import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import os

# ---------------------
# Data Preparation Function
# ---------------------

def getData(data_name, sparsity="dense", error=False):
    """
    Load pre-simulated datasets for training and testing.
    
    Args:
        data_name (str): Name of the dataset.
        sparsity (str): Sparsity setting, either 'dense' or 'sparse'.
        error (bool): Whether to include error in the dataset.

    Returns:
        tuple: (X, T, sparsity_error_folder), where:
            X (pd.DataFrame): Observed data matrix.
            T (pd.DataFrame): Corresponding time points.
            sparsity_error_folder (str): Path to the folder based on settings.
    
    Raises:
        Exception: If the required files are not found.
    """
    # Construct folder path based on dataset name, sparsity, and error settings
    sparsity_error_folder = "/" + ((sparsity + "/w_error" if error else sparsity + "/wo_error")
                                   if data_name != "UK" else sparsity) if data_name != "Framingham" else ""
    folder_path = "../Data/" + data_name + sparsity_error_folder

    # Check if required files exist and load them
    if os.path.exists(folder_path + "/X_obs.csv") and os.path.exists(folder_path + "/T_obs.csv"):
        X = pd.read_csv(folder_path + "/X_obs.csv", header=None)
        T = pd.read_csv(folder_path + "/T_obs.csv", header=None)
    else:
        raise Exception("X_obs.csv or T_obs.csv is not found in " + os.path.abspath(folder_path) +
                        "\n Please use R to simulate the dataset.")
    
    return X, T, sparsity_error_folder

# ---------------------
# DataLoader Class
# ---------------------

class DataLoader:
    """
    DataLoader class to manage training, validation, and testing datasets.
    Handles preprocessing, train/validation/test splits, and batch generation.
    """
    def __init__(self, X, T, dataloader_settings, random_seed=321):
        """
        Initialize the DataLoader with preprocessing and splitting logic.

        Args:
            X (pd.DataFrame): Observed data matrix.
            T (pd.DataFrame): Time points corresponding to X.
            dataloader_settings (dict): Contains parameters like batch size and device.
            random_seed (int): Random seed for reproducibility.
        """
        np.random.seed(random_seed)  # Set random seed for reproducibility
        self.device = dataloader_settings["device"]  # Device (CPU or GPU)
        d, split = dataloader_settings["d"], dataloader_settings["split"]  # Dimensions and split settings

        # Number of observations per sample
        num_obs = np.array(X.iloc[:, 0], dtype=int)
        X = X.iloc[:, 1:]  # Remove the first column (used as observation count)
        self.n, self.m = X.shape  # Number of samples (n) and time points (m)

        # Replace NaN values in X and T with 0
        X.replace(np.nan, 0, inplace=True)
        T.replace(np.nan, 0, inplace=True)

        # Total number of unique time points in T
        self.tptsTesting = len(np.unique(T))

        # Create probabilistic masks for non-iid settings
        full_prob_list = np.zeros((self.n, max(num_obs) * 2 + 1))
        for i in range(self.n):
            full_prob_list[i, :num_obs[i] * 2 + 1] = norm.pdf(
                x=np.array(range(-num_obs[i], num_obs[i] + 1)),
                loc=0,
                scale=(T.iloc[i, num_obs[i] - 1] - T.iloc[i, 0]) * 15
            )

        # Encode time points using sine and cosine
        encT = np.zeros([self.n, d, self.m])
        for i in range(int(d / 2)):
            encT[:, 2 * i, :] = np.sin(10 ** (-4 * (i + 1) / d) * T * (self.tptsTesting - 1))
            encT[:, 2 * i + 1, :] = np.cos(10 ** (-4 * (i + 1) / d) * T * (self.tptsTesting - 1))

        # Compute dataset splits
        self.batch_size = dataloader_settings["batch_size"]
        self.train_n = round(self.n * split[0] / sum(split))
        self.valid_n = round(self.n * split[1] / sum(split))
        self.test_n = self.n - self.train_n - self.valid_n

        # Split data into train, validation, and test sets
        self.train_X = torch.Tensor(np.array(X.iloc[:self.train_n, :])).to(self.device).unsqueeze(1)
        self.train_T = torch.Tensor(np.array(T.iloc[:self.train_n, :])).to(self.device).unsqueeze(1)
        self.train_encT = torch.Tensor(encT[:self.train_n, :, :]).to(self.device)
        self.train_O = num_obs[:self.train_n]
        self.train_prob_list = torch.Tensor(full_prob_list[:self.train_n, :]).to(self.device)

        self.valid_X = torch.Tensor(np.array(X.iloc[self.train_n:(self.train_n + self.valid_n), :])).to(self.device).unsqueeze(1)
        self.valid_T = torch.Tensor(np.array(T.iloc[self.train_n:(self.train_n + self.valid_n), :])).to(self.device).unsqueeze(1)
        self.valid_encT = torch.Tensor(encT[self.train_n:(self.train_n + self.valid_n), :, :]).to(self.device)
        self.valid_O = num_obs[self.train_n:(self.train_n + self.valid_n)]
        self.valid_prob_list = torch.Tensor(full_prob_list[self.train_n:(self.train_n + self.valid_n), :]).to(self.device)

        self.test_X = torch.Tensor(np.array(X.iloc[(self.train_n + self.valid_n):, :])).to(self.device).unsqueeze(1)
        self.test_T = torch.Tensor(np.array(T.iloc[(self.train_n + self.valid_n):, :])).to(self.device).unsqueeze(1)
        self.test_encT = torch.Tensor(encT[(self.train_n + self.valid_n):, :, :]).to(self.device)
        self.test_O = num_obs[(self.train_n + self.valid_n):]
        self.test_prob_list = torch.Tensor(full_prob_list[(self.train_n + self.valid_n):]).to(self.device)

        # Maximum and minimum values for scaling
        self.max_X, self.min_X = torch.max(self.train_X).to(self.device), torch.min(self.train_X).to(self.device)

        # Encode full range of time points for testing
        full_T = np.linspace(0, 1, len(np.unique(T)))
        enc_full_T = np.zeros([1, d + 1, len(full_T)])
        enc_full_T[:, 0, :] = full_T
        for i in range(int(d / 2)):
            enc_full_T[:, 2 * i + 1, :] = np.sin(10 ** (-4 * (i + 1) / d) * full_T * (self.tptsTesting - 1))
            enc_full_T[:, 2 * i + 2, :] = np.cos(10 ** (-4 * (i + 1) / d) * full_T * (self.tptsTesting - 1))

        self.tptsTraining = full_T.shape[-1]
        self.full_T = torch.Tensor(enc_full_T).to(self.device)

    def shuffle(self):
        """
        Shuffle the training dataset for each epoch.
        """
        new_order = np.arange(self.train_X.shape[0])
        np.random.shuffle(new_order)
        self.train_X = self.train_X[new_order]
        self.train_T = self.train_T[new_order]
        self.train_encT = self.train_encT[new_order]
        self.train_O = self.train_O[new_order]
        self.train_prob_list = self.train_prob_list[new_order]

    def _batch_generator(self, X, T, encT, O, n, prob_list):
        """
        Internal generator for batching data.

        Args:
            X (Tensor): Data matrix.
            T (Tensor): Time points.
            encT (Tensor): Encoded time points.
            O (np.array): Observations count per sample.
            n (int): Total sample size.
            prob_list (Tensor): Probabilistic mask.

        Yields:
            tuple: Masked data and associated tensors for training/testing.
        """
        def generator_func():
            N = round(np.ceil(n / self.batch_size))  # Number of batches per epoch
            treated_as_noniid = (np.random.randint(low=0, high=2, size=N) == 1)  # Non-iid flag
            y_t = torch.repeat_interleave(self.full_T, self.batch_size, axis=0)

            for i in range(N):
                # Extract batch-specific data
                x = X[(i * self.batch_size):((i + 1) * self.batch_size)]
                t = T[(i * self.batch_size):((i + 1) * self.batch_size)]
                enct = encT[(i * self.batch_size):((i + 1) * self.batch_size)]
                obs = O[(i * self.batch_size):((i + 1) * self.batch_size)]
                prob_list_now = prob_list[(i * self.batch_size):((i + 1) * self.batch_size)]
                src = torch.cat([x, t, enct], dim = 1)
                batch_size_now = len(obs)
                
                y = torch.zeros(batch_size_now, self.full_T.shape[-1], device = self.device)
                d_mask = torch.zeros(batch_size_now, self.full_T.shape[-1], device = self.device, dtype = int)
                if batch_size_now != self.batch_size:
                    y_t = torch.repeat_interleave(self.full_T, batch_size_now, axis = 0)
                index_mat = torch.round(t * (self.tptsTraining - 1)).squeeze(1).long()
                
                # mask related: 1 = seen; 0 = unseen
                enc_mask_size = np.random.randint(low = 1, high = obs/2 + 1, size = batch_size_now)
                e_mask = (t > 0).int().to(self.device).clone().detach().squeeze(1)
                e_mask[:, 0] = 1
                    
                emb_mask = e_mask.clone()
                if treated_as_noniid[i]:
                    center_vec = np.random.randint(low = 0, high = obs, size = batch_size_now)
                
                for j in range(batch_size_now):
                    index_col = index_mat[j, range(obs[j])]
                    y[j, index_col] = x[j, 0, range(obs[j])]
                    d_mask[j, index_col] = 1
                    
                    if treated_as_noniid[i]:
                        center = center_vec[j]
                        prob = prob_list_now[j, (obs[j] - center + 1):(2 * obs[j] - center + 1)]
                        unmasked_idx = torch.multinomial(prob, obs[j] - enc_mask_size[j])
                        e_mask[j, unmasked_idx] = 2
                    else:
                        masked_idx = np.random.randint(low = 0, high = obs[j], size = enc_mask_size[j])
                        e_mask[j, masked_idx] = 0
                    
                if treated_as_noniid[i]:
                    e_mask = (e_mask == 2).int()
                
                yield emb_mask, e_mask, d_mask, src, y_t, y

        return generator_func()

    def get_train_batch(self):
        """Generate batches for the training set."""
        return self._batch_generator(self.train_X, self.train_T, self.train_encT, self.train_O, self.train_n, self.train_prob_list)

    def get_valid_batch(self):
        """Generate batches for the validation set."""
        return self._batch_generator(self.valid_X, self.valid_T, self.valid_encT, self.valid_O, self.valid_n, self.valid_prob_list)

    def get_test_batch(self):
        """Generate batches for the test set."""
        return self._batch_generator(self.test_X, self.test_T, self.test_encT, self.test_O, self.test_n, self.test_prob_list)
