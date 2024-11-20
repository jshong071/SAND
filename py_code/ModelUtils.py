# Essential imports
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
        self.lin3 = nn.Linear(self.hidden, f_out)

        # Normalization layers for stability
        self.norm1 = Norm(self.hidden, axis=-1)
        self.norm2 = Norm(self.hidden, axis=-1)
        self.norm3 = Norm(f_out, axis=-1)

        # Dropout layers for regularization
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

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
            x_ = x_ + F.relu(self.lin2(x_))  # First residual connection
            x_ = self.norm1(x_)
            x_ = x_ + F.relu(self.lin2(self.drop1(x_)))  # Second residual connection
            x_ = self.norm2(x_)
            x_ = F.relu(self.lin3(self.drop2(x_)))
        else:
            # Apply GELU activation and residual connections
            x_ = F.gelu(self.lin1(x))
            x_ = x_ + F.gelu(self.lin2(x_))  # First residual connection
            x_ = self.norm1(x_)
            x_ = x_ + F.gelu(self.lin2(self.drop1(x_)))  # Second residual connection
            x_ = self.norm2(x_)
            x_ = F.gelu(self.lin3(self.drop2(x_)))

        # Final normalization
        return self.norm3(x_)
