# Transformer with SAND for functional data imputation
# ========================================
# 1. The shape of the tensor passed across all modules is kept as (batch, d, seq_len)
# 2. Except for the attention module, outputs are normalized across modules, so input normalization is unnecessary.
# ========================================

import torch
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from Encoder import TransformerEncoder  # Custom Transformer encoder module
from Decoder import TransformerDecoder  # Custom Transformer decoder module
from SAND import SAND                  # Self-Attention on Derivatives (SAND) module
from ModelUtils import FeedForward, compute_loss  # Auxiliary utility functions

class Transformer(nn.Module):
    """
    Transformer model augmented with the SAND module for smooth functional imputation.
    """
    def __init__(self, model_settings):
        super().__init__()
        torch.manual_seed(321)  # Set seed for reproducibility
        
        # Define model-specific parameters
        model_settings["num_q"] = model_settings["num_k"] = model_settings["num_v"] = model_settings["num_hiddens"]
        self.model_settings = model_settings
        self.dataloader_settings = model_settings["dataloader_settings"]
        
        # Core Transformer modules
        self.encoder = TransformerEncoder(model_settings)  # Encoder for feature extraction
        self.decoder = TransformerDecoder(model_settings)  # Decoder for imputation
        
        # Output layers for final prediction
        self.linear = nn.Linear(model_settings["num_hiddens"], 1)  # Linear transformation for predictions
        self.sigmoid = nn.Sigmoid()  # Ensures output scaling

        # Scaling parameters for imputations
        self.alpha_org = model_settings["max_X"] - model_settings["min_X"]
        self.beta_org = model_settings["min_X"]

        # Initialize SAND-specific settings
        model_settings2 = model_settings.copy()
        model_settings2["num_hiddens"] = model_settings["f_in"][0]
        model_settings2["dropout"] = 0.05  # Separate dropout rate for SAND
        self.SAND = SAND(model_settings2)

    def forward(self, x, y_t, e_m_mh, d_m_mh, d_m, iteration=0, TAs_position=None, isTraining=True):
        """
        Forward pass of the Transformer model.
        Args:
            x (Tensor): Input tensor of observed features.
            y_t (Tensor): Target tensor for decoder input.
            e_m_mh (Tensor): Encoder masks for multi-head attention.
            d_m_mh (Tensor): Decoder masks for multi-head attention.
            d_m (Tensor): Decoder mask for observed points.
            iteration (int): Current training iteration for tracking.
            TAs_position (Tensor): Positions of time anchors.
            isTraining (bool): Whether the model is in training mode.

        Returns:
            list: Smooth imputations (SAND) and raw imputations (vanilla Transformer).
        """
        # Encoder forward pass
        e_output = self.encoder(x, e_m_mh)

        # Decoder forward pass
        d_output = self.decoder(y_t, e_output, e_m_mh, d_m_mh)

        # Output from the vanilla Transformer
        org = self.linear(d_output)
        org_detach = org.detach().clone()[:, :, 0].unsqueeze(-1)

        # SAND-enhanced smooth output
        smooth = self.SAND(org_detach, y_t, d_m, iteration, TAs_position, isTraining)
        
        return [smooth, self.sigmoid(org) * self.alpha_org + self.beta_org]

    def StartTraining(self, dataLoader, optimizer=None, save_model_every=100, start_from_k=0, verbose=False):
        """
        Start the training loop for the Transformer model.

        Args:
            dataLoader (DataLoader): Data loader for training and validation batches.
            optimizer (Optimizer): Optimizer for gradient updates.
            save_model_every (int): Frequency (in epochs) for saving checkpoints.
            start_from_k (int): Epoch to resume training from.
            verbose (bool): Whether to print training progress.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.model_settings["lr"])

        # Prepare model settings and directories
        model_settings = self.model_settings
        dataloader_settings = self.dataloader_settings
        sparsity_error_folder = model_settings["sparsity_error_folder"]
        device = model_settings["device"]
        data_name = dataloader_settings["data_name"]

        # Initialize tracking variables for losses
        loss_train_history = []
        loss_valid_history = []
        min_valid_loss = sys.maxsize
        
        scaler = GradScaler()
        # Training loop
        for k in range(start_from_k, model_settings["max_epoch"]):
            # Save checkpoints periodically
            if k % save_model_every == 0:
                checkpoint = {
                    "model": self.model_settings,
                    "state_dict": self.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                torch.save(checkpoint, f"../Checkpoints/{data_name}{sparsity_error_folder}/ckpts_{k}.pth")
            
            # Reset losses for this epoch
            loss_train = []
            loss_valid = []

            # Shuffle training data for each epoch
            dataLoader.shuffle()

            # Switch model to training mode
            self = self.train()
            
            # Process training batches
            for i, (emb_mask, e_m, d_m, x, y_t, y) in enumerate(dataLoader.get_train_batch()):
                optimizer.zero_grad()
                with autocast():
                    # Prepare masks for multi-head attention
                    e_m_mh = torch.repeat_interleave(e_m, model_settings["num_heads"], dim=0)
                    d_m_mh = torch.repeat_interleave(d_m, model_settings["num_heads"], dim=0)
    
                    # Forward pass and compute loss
                    out = self.forward(x, y_t, e_m_mh, d_m_mh, d_m, iteration=k)
                    loss = compute_loss(out, y, d_m, k, isTraining=True)
                    loss_train.append(loss.item())

                # Backpropagation
                scaler.scale(loss).backward()  # Scales loss for mixed precision
                scaler.step(optimizer)
                scaler.update()
            
            # Average training loss for this epoch
            avg_loss_train = np.mean(loss_train)
            loss_train_history.append(avg_loss_train)

            # Validation phase
            with torch.no_grad():
                self = self.eval()
                for emb_mask, e_m, d_m, x, y_t, y in dataLoader.get_valid_batch():
                    e_m_mh = torch.repeat_interleave(e_m, model_settings["num_heads"], dim=0)
                    d_m_mh = torch.repeat_interleave(d_m, model_settings["num_heads"], dim=0)
                    valid_y = self.forward(x, y_t, e_m_mh, d_m_mh, d_m, isTraining=False)
                    loss_valid_tmp = compute_loss(valid_y, y, d_m, k, isTraining=False)
                    loss_valid.append(loss_valid_tmp.item())

            # Update the best model if validation loss improves
            if np.mean(loss_valid) < min_valid_loss:
                checkpoint_best = {
                    "model": self.model_settings,
                    "state_dict": self.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                torch.save(checkpoint_best, f"../Checkpoints/{data_name}{sparsity_error_folder}/best_ckpts.pth")
                min_valid_loss = np.mean(loss_valid)

            # Append validation loss
            loss_valid_history.append(np.mean(loss_valid))

            # Print progress if verbose
            if verbose:
                if np.mean(loss_valid) == min_valid_loss:
                    print(f"The best model is at epoch {k}: Training loss = {loss_train_history[-1]}, Validation loss = {loss_valid_history[-1]}", flush=True)
                if k < 20 or k % 50 == 0:
                    print(f"Epoch {k}: Training loss = {loss_train_history[-1]}, Validation loss = {loss_valid_history[-1]}", flush=True)

def GetImputation(X_obs, T_obs, data_name="HighDim_E", sparsity="dense", error=False):
    """
    Perform functional data imputation using a trained Transformer model with SAND.
    
    Args:
        X_obs (DataFrame): Observed data matrix (n_samples, d_features + 1), where the first column is the mask.
        T_obs (DataFrame): Observed time points (n_samples, seq_len).
        data_name (str): Name of the dataset.
        sparsity (str): Sparsity type ("dense" or "sparse").
        error (bool): Whether the data contains error terms.

    Returns:
        list: Two arrays:
            - Smooth imputations from the SAND-enhanced Transformer.
            - Raw imputations from the vanilla Transformer.
    """
    # Verify input dimensions
    assert X_obs.shape[0] == T_obs.shape[0], "Mismatch in number of samples between X_obs and T_obs."
    assert X_obs.shape[1] - 1 == T_obs.shape[1], "Mismatch in feature dimensions between X_obs and T_obs."

    # Load the trained model checkpoint
    device = torch.device("cpu")
    sparsity_error_folder = (
        "/" + ((sparsity + "/w_error" if error else sparsity + "/wo_error") if data_name != "UK" else sparsity)
        if data_name != "Framingham"
        else ""
    )
    checkpoint_path = f"../Checkpoints/{data_name}{sparsity_error_folder}/best_ckpts.pth"
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize the Transformer model and load the checkpoint
    model = Transformer(checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.model_settings["device"] = device

    # Disable gradient computation for inference
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model = model.to(device)
    model = model.eval()

    # Extract model settings
    num_heads = model.model_settings["num_heads"]
    L = model.model_settings["tptsTraining"]  # Number of training time points
    t_true = np.linspace(0, 1, L)  # Equally spaced time points
    d = model.model_settings["f_in"][0] - 2  # Dimensionality excluding time and auxiliary features

    # Generate auxiliary time features for the decoder
    d_T = np.zeros([1, d + 1, L])
    d_T[:, 0, :] = t_true  # Time component
    for i in range(int(d / 2)):
        d_T[0, 2 * i + 1, :] = np.sin(10 ** (-4 * (i + 1) / d) * t_true * (L - 1))
        d_T[0, 2 * i + 2, :] = np.cos(10 ** (-4 * (i + 1) / d) * t_true * (L - 1))

    d_T = torch.Tensor(d_T).to(device)
    d_m_mh = torch.Tensor([[1] * L] * num_heads).to(device)  # Multi-head mask for the decoder
    d_m = torch.Tensor([[1] * L]).to(device)  # Single-head mask for the decoder

    # Initialize predictions
    org_pred = []  # Raw predictions from vanilla Transformer
    smooth_pred = []  # Smooth predictions from SAND

    # Define weights for probabilistic smoothing
    prob_list = torch.Tensor(
        np.array([np.exp((1 - torch.abs(torch.Tensor(range(-L, L + 1)) / L)) * 50).numpy()])
    )
    prob_list = prob_list / prob_list.sum(1, keepdim=True) + (2 ** -32)
    prob_list = prob_list / prob_list.sum(1, keepdim=True)

    # Extract observed mask and input features
    M_obs = np.array(X_obs.iloc[:, 0], dtype=int)  # Observation mask
    X_obs = X_obs.iloc[:, 1:]  # Observed feature matrix
    n, m = X_obs.shape  # Number of samples and sequence length

    # Iterate over each sample for imputation
    for i in range(X_obs.shape[0]):
        # Create source mask (1 for observed, 0 for missing values)
        src_mask = [1] * m
        src_mask[M_obs[i]:] = [0] * (m - M_obs[i])

        # Extract feature and time data for the current sample
        x = X_obs.iloc[i, :]
        t = np.array(T_obs.iloc[i, :])

        # Create encoder inputs
        e_XT = np.zeros([1, d + 1, len(t)])
        e_XT[:, 0, :] = t  # Time component
        for j in range(int(d / 2)):
            e_XT[0, 2 * j + 1, :] = np.sin(10 ** (-4 * (j + 1) / d) * t * (L - 1))
            e_XT[0, 2 * j + 2, :] = np.cos(10 ** (-4 * (j + 1) / d) * t * (L - 1))

        e_X = torch.Tensor(np.concatenate((np.array([[x]]), e_XT), axis=1))  # Combine observed features and time
        e_m = torch.Tensor(src_mask).reshape(1, -1)  # Encoder mask
        e_m_mh = torch.repeat_interleave(e_m, num_heads, dim=0).to(device)

        # Calculate time anchor positions
        TAs_position = np.array(e_X[0, 1, :] * (L - 1), dtype=int)
        TAs_position = torch.Tensor(TAs_position[np.diff(np.concatenate(([-1], TAs_position))) > 0]).unsqueeze(-1).to(device)

        # Forward pass through the model
        [smooth, org] = model.forward(e_X.to(device), d_T, e_m_mh, d_m_mh, d_m, TAs_position=TAs_position)

        # Weight smoothing probabilities
        weight = torch.zeros(smooth.size()[0], smooth.size()[1], 1)
        for j, pos in enumerate(TAs_position.squeeze(-1).long()):
            weight[j, :, 0] = prob_list[0][(L - pos + 1):(2 * L - pos + 1)]

        weight = (weight / weight.sum(0)).to(device)
        smooth_prob = torch.sum(smooth * weight, dim=0)

        # Store predictions
        smooth_pred.append(smooth_prob.cpu().numpy())
        org_pred.append(org.squeeze(0).cpu().numpy())

    return [np.array(smooth_pred).squeeze(-1), np.array(org_pred).squeeze(-1)]
