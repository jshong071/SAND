# Standard library imports
import os
import re
from statistics import median

# Scientific computing and data analysis
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Deep learning framework
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_loss(y_pred, y, mask, iteration, isTraining=True):
    """
    Compute the loss for model predictions.

    Args:
        y_pred (list): A list containing [smooth predictions, original predictions].
        y (Tensor): Ground truth values.
        mask (Tensor): Mask to indicate observed points.
        iteration (int): Current iteration of training.
        isTraining (bool): Whether the model is in training mode.

    Returns:
        Tensor: Computed loss based on the mean squared error (MSE).
    """
    # Unpack predictions: smooth (SAND) and original (vanilla Transformer)
    [smooth, original] = y_pred

    # Compute Smooth MSE
    smoothMSE = torch.sqrt(torch.sum((smooth[:, :, 0] - y) ** 2 * mask) / torch.sum(mask))
    smoothMSE_train = smoothMSE.clone()

    # Compute Original MSE
    originalMSE = torch.sqrt(torch.sum((original[:, :, 0] - y) ** 2 * mask) / torch.sum(mask))
    originalMSE_train = originalMSE.clone()

    # Weighted loss during training
    if isTraining:
        return originalMSE_train * 0.75 + smoothMSE_train * 0.25
    else:
        return smoothMSE


class Norm(nn.Module):
    """
    Custom normalization layer with learnable parameters.

    Args:
        d (int): Dimensionality of the input.
        axis (int): Axis along which normalization is applied.
        eps (float): Small constant to prevent division by zero.
    """
    def __init__(self, d, axis=-2, eps=1e-6):
        super().__init__()
        self.d = d
        self.axis = axis

        # Learnable parameters for scaling (alpha) and shifting (bias)
        if axis == -2:
            self.alpha = nn.Parameter(torch.randn((d, 1)))
            self.bias = nn.Parameter(torch.randn((d, 1)))
        else:
            self.alpha = nn.Parameter(torch.ones(d))
            self.bias = nn.Parameter(torch.zeros(d))

        self.eps = eps

    def forward(self, x):
        """
        Forward pass for normalization.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Normalized output.
        """
        avg = x.mean(dim=self.axis, keepdim=True) if self.d > 1 else 0
        std = x.std(dim=self.axis, keepdim=True) + self.eps if self.d > 1 else 1
        norm = self.alpha * (x - avg) / std + self.bias
        return norm


class FeedForward(nn.Module):
    """
    Feedforward network with normalization and dropout, supporting residual connections.

    Args:
        f_in (int): Input feature dimension.
        f_out (int): Output feature dimension.
        hidden (int): Hidden layer size.
        dropout (float): Dropout rate.
    """
    def __init__(self, f_in, f_out, hidden=64, dropout=0.1):
        super().__init__()
        self.hidden = hidden

        # Linear layers for transformations
        self.lin1 = nn.Linear(f_in, self.hidden)
        self.lin2 = nn.Linear(self.hidden, self.hidden)
        self.lin3 = nn.Linear(self.hidden, self.hidden)
        self.lin4 = nn.Linear(self.hidden, f_out)

        # Normalization layers for stability
        self.norm1 = Norm(self.hidden, axis=-1)
        self.norm2 = Norm(self.hidden, axis=-1)
        self.norm3 = Norm(self.hidden, axis=-1)
        self.norm4 = Norm(f_out, axis=-1)

        # Dropout layers for regularization
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, useReLU=True):
        """
        Forward pass for the feedforward network.

        Args:
            x (Tensor): Input tensor.
            useReLU (bool): Whether to use ReLU (True) or GELU (False) as the activation function.

        Returns:
            Tensor: Transformed and normalized output.
        """
        if useReLU:
            # Apply ReLU activation and residual connections
            x_ = F.relu(self.lin1(x))
            x_ = self.norm1(x_)
            x_ = x_ + F.relu(self.lin2(self.drop1(x_)))  # First residual connection
            x_ = self.norm2(x_)
            x_ = x_ + F.relu(self.lin3(self.drop2(x_)))  # Second residual connection
            x_ = self.norm3(x_)
            x_ = F.relu(self.lin4(self.drop3(x_)))
        else:
            # Apply GELU activation and residual connections
            x_ = F.gelu(self.lin1(x))
            x_ = self.norm1(x_)
            x_ = x_ + F.gelu(self.lin2(self.drop1(x_)))  # First residual connection
            x_ = self.norm2(x_)
            x_ = x_ + F.gelu(self.lin3(self.drop2(x_)))  # Second residual connection
            x_ = self.norm3(x_)
            x_ = F.gelu(self.lin4(self.drop3(x_)))

        # Final normalization
        return self.norm4(x_)

def plot_imputations(i, X_obs, T_obs, VT_imp=None, SAND_imp=None, X_den=None, figsize=(10, 6)):
    """
    Plot observed data points, imputations, and ground truth for a given index.
    
    Args:
        i (int): Index of the sequence to plot
        X_obs (pd.DataFrame): Observed values DataFrame
        T_obs (pd.DataFrame): Observed timepoints DataFrame
        VT_imp (np.ndarray, optional): Vanilla Transformer imputations
        SAND_imp (np.ndarray, optional): SAND imputations
        X_den (np.ndarray, optional): Ground truth values
        figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Get observed data for index i
    n_obs = int(X_obs.iloc[i, 0])  # Number of observations
    x_vals = X_obs.iloc[i, 1:n_obs+1].values  # Observed values
    t_vals = T_obs.iloc[i, :n_obs].values  # Observed timepoints
    
    # Plot observed data points
    plt.scatter(t_vals, x_vals, color='black', label='Observations', zorder=5)
    
    # If imputations are provided, plot them
    if VT_imp is not None or SAND_imp is not None or X_den is not None:
        # Generate imputation grid
        unique_times = np.sort(np.unique(T_obs[T_obs != 0].values))
        L = len(unique_times)
        t_grid = np.linspace(0, 1, L)
    
    # Plot ground truth if provided
    if X_den is not None:
        plt.plot(t_grid, X_den[i], label='Ground Truth', 
                linestyle=':', color='black', alpha=0.7)
    
    # Plot Vanilla Transformer imputations if provided
    if VT_imp is not None:
        plt.plot(t_grid, VT_imp[i], label='Vanilla Transformer', 
                linestyle='--', color='blue', alpha=0.7)
    
    # Plot SAND imputations if provided
    if SAND_imp is not None:
        plt.plot(t_grid, SAND_imp[i], label='SAND',
                linestyle='-', color='red', alpha=0.7)
    
    # Customize plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Sequence {i}: Observed Data and Imputations')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set reasonable axis limits
    all_x_values = [x_vals]
    if VT_imp is not None:
        all_x_values.append(VT_imp[i])
    if SAND_imp is not None:
        all_x_values.append(SAND_imp[i])
    if X_den is not None:
        all_x_values.append(X_den[i])
    
    x_min = min(np.concatenate(all_x_values))
    x_max = max(np.concatenate(all_x_values))
    y_padding = (x_max - x_min) * 0.1
    
    plt.ylim(x_min - y_padding, x_max + y_padding)
    plt.xlim(-0.02, 1.02)  # Slight padding on time axis
    
    plt.show()

def find_largest_number_in_filenames(folder_path):
    """
    Find the largest number in filenames matching pattern 'XX_number.pth'
    where the file size is larger than half of the median file size.
    
    Args:
        folder_path (str): Path to the folder containing the files
        
    Returns:
        int: Largest number found in filenames meeting the criteria, or 0 if no matching files
        
    Raises:
        FileNotFoundError: If the folder_path doesn't exist
    """
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
        
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Store file information (number and size) for matching files
    file_info = []
    for filename in files:
        if filename.endswith('.pth'):
            match = re.search(r'_(\d+)\.pth$', filename)
            if match:
                try:
                    file_path = os.path.join(folder_path, filename)
                    file_size = os.path.getsize(file_path)
                    number = int(match.group(1))
                    file_info.append({
                        'number': number,
                        'size': file_size,
                        'filename': filename
                    })
                except (OSError, ValueError) as e:
                    print(f"Warning: Error processing file {filename}: {e}")
                    continue
    
    # Return 0 if no matching files found
    if not file_info:
        return 0
    
    try:
        # Calculate threshold (half of median file size)
        median_size = median([f['size'] for f in file_info])
        size_threshold = median_size / 2
        
        # Filter for files larger than threshold and get their numbers
        numbers = [f['number'] for f in file_info if f['size'] > size_threshold]
        
        # Return the largest number if found, 0 otherwise
        return max(numbers) if numbers else 0
        
    except Exception as e:
        print(f"Warning: Error calculating result: {e}")
        return 0
