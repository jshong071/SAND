import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ModelUtils import FeedForward  # Custom FeedForward utility

def transpose_qkv(x, num_heads):
    """
    Transpose the query, key, or value tensor for parallel computation across multiple attention heads.

    Args:
        x (Tensor): Input tensor of shape (batch, seq_len, num_hidden).
        num_heads (int): Number of attention heads.

    Returns:
        Tensor: Transposed tensor of shape (batch * num_heads, seq_len, num_hidden / num_heads).
    """
    if num_heads == 1:
        return x  # No transposition needed for single head
    else:
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)  # Split hidden dimensions into multiple heads
        x = x.transpose(1, 2)  # Swap head and sequence length dimensions
        return x.reshape(-1, x.shape[2], x.shape[3])  # Flatten batch and head dimensions

def transpose_output(x, num_heads):
    """
    Reverse the operation of `transpose_qkv` to recombine attention heads.

    Args:
        x (Tensor): Input tensor of shape (batch * num_heads, seq_len, num_hidden / num_heads).
        num_heads (int): Number of attention heads.

    Returns:
        Tensor: Recombined tensor of shape (batch, seq_len, num_hidden).
    """
    if num_heads == 1:
        return x  # No transposition needed for single head
    else:
        x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])  # Separate batch and head dimensions
        x = x.transpose(1, 2)  # Swap head and sequence length dimensions back
        return x.reshape(x.shape[0], x.shape[1], -1)  # Flatten head dimensions into hidden dimensions

def _attention(q, k, v, d_k, mask_multihead, Dropout=None, output_structure=None):
    """
    Compute scaled dot-product attention.

    Args:
        q (Tensor): Query tensor of shape (batch, seq_len, d_k).
        k (Tensor): Key tensor of shape (batch, seq_len, d_k).
        v (Tensor): Value tensor of shape (batch, seq_len, d_k).
        d_k (int): Dimensionality of the key vectors.
        mask_multihead (Tensor): Mask tensor of shape (batch, seq_len) or None.
        Dropout (nn.Dropout, optional): Dropout layer applied to attention scores.
        output_structure (str, optional): Custom output structure (e.g., for SAND).

    Returns:
        Tensor: Output tensor after applying attention, shape (batch, seq_len, d_k).
    """
    # Compute scaled dot-product scores
    scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)  # Shape: (batch, seq_len, seq_len)
    
    if output_structure is None:
        # Apply mask for standard attention
        if mask_multihead is not None:
            mask_multihead = mask_multihead.unsqueeze(1)  # Expand mask for multi-head
            scores = scores.masked_fill(mask_multihead == 0, -1e4)  # Mask out invalid positions
        scores = F.softmax(scores, dim=-1)  # Normalize scores across sequence length
    else:
        # Special handling for custom output structures (e.g., SAND)
        scores /= v.shape[1]  # Scale scores by sequence length

    # Apply dropout if specified
    scores = Dropout(scores) if Dropout is not None else scores
    
    # Compute attention-weighted values
    return torch.matmul(scores, v)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_settings (dict): Dictionary of model settings including:
            - num_q, num_k, num_v: Dimensions of queries, keys, and values.
            - num_hiddens: Total hidden dimensionality.
            - num_heads: Number of attention heads.
            - dropout: Dropout rate.
        output_structure (str, optional): Custom output structure, defaults to None.
    """
    def __init__(self, model_settings, output_structure=None):
        super(MultiHeadAttention, self).__init__()
        
        # Extract settings from the model configuration
        num_q = model_settings["num_q"]
        num_k = model_settings["num_k"]
        num_v = model_settings["num_v"]
        num_hiddens = model_settings["num_hiddens"]
        dropout = model_settings["dropout"]
        num_heads = model_settings["num_heads"] if output_structure is None else 1
        
        # Compute dimensions for each head
        self.f_out = num_q // num_heads
        self.num_heads = num_heads
        self.output_structure = output_structure
        
        # Linear layers for queries, keys, and values
        self.q_lin = nn.Linear(num_hiddens, num_q, bias=True)
        self.k_lin = nn.Linear(num_hiddens, num_k, bias=True)
        self.v_lin = nn.Linear(num_hiddens, num_v, bias=True)
        
        # FeedForward layers for non-linear transformations
        self.q_ff = FeedForward(f_in=self.f_out, f_out=self.f_out, dropout=dropout)
        self.k_ff = FeedForward(f_in=self.f_out, f_out=self.f_out, dropout=dropout)
        self.v_ff = FeedForward(f_in=self.f_out, f_out=self.f_out, dropout=dropout)
        
        # Output transformation layer
        if output_structure is None:
            self.out = nn.Linear(self.f_out, self.f_out, bias=True)
        else:
            self.out = nn.Linear(self.f_out, 1, bias=True)  # Custom output for specific structures
        
        # Dropout layer
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            q (Tensor): Query tensor of shape (batch, num_hidden, seq_len).
            k (Tensor): Key tensor of shape (batch, num_hidden, seq_len).
            v (Tensor): Value tensor of shape (batch, num_hidden, seq_len).
            mask (Tensor, optional): Mask tensor for invalid positions.

        Returns:
            Tensor: Output tensor of shape (batch, seq_len, num_hidden).
        """
        # Apply linear transformations and transpose for multi-head processing
        Q = self.q_ff(transpose_qkv(self.q_lin(q), self.num_heads))
        K = self.k_ff(transpose_qkv(self.k_lin(k), self.num_heads))
        V = self.v_ff(transpose_qkv(self.v_lin(v), self.num_heads))
        
        # Compute attention
        output = _attention(Q, K, V, self.f_out, mask, self.drop, self.output_structure)
        
        # Apply final linear transformation and combine heads
        output = self.out(output)
        return transpose_output(output, self.num_heads)
