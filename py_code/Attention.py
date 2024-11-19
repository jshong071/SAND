import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ModelUtils import FeedForward

# class Object(object):
#     pass
# self = Object()
# self = model.outblock.outblks.block0.eval()

def transpose_qkv(x, num_heads):
    """Transposition for parallel computation of multiple attention heads"""
    if num_heads == 1:
        return x
    else:
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
        x = x.transpose(1, 2)
        return x.reshape(-1, x.shape[2], x.shape[3])

def transpose_output(x, num_heads):
    """Reverse the operation of `transpose_qkv`"""
    if num_heads == 1:
        return x
    else:
        x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
        x = x.transpose(1, 2)
        return x.reshape(x.shape[0], x.shape[1], -1)

def _attention(q, k, v, d_k, mask_multihead, Dropout = None, output_structure = None):
    # input q, v, k size: (batch, seq_len, d)
    # input mask size   : (batch, seq_len)
    # q, k, v, d_k = Q, K, V, self.f_out
    # mask_multihead = e_m_mh
    # mask_multihead = d_m
    scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k) # scores size : (batch, seq_len, seq_len)
    if output_structure == None:
        if mask_multihead != None:
            mask_multihead = mask_multihead.unsqueeze(1)
            scores = scores.masked_fill(mask_multihead == 0, -1e9)
        scores = F.softmax(scores, dim = -1)
    else: ## SAND
        scores /= v.shape[1] # scores size : (batch, seq_len, seq_len)
    scores = Dropout(scores) if Dropout != None else scores
    return torch.matmul(scores, v)

class MultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, model_settings, output_structure = None):
        super(MultiHeadAttention, self).__init__()
        # output_structure = "SelfAtt" # "SelfAtt" None
        num_q, num_k, num_v, num_hiddens, dropout = model_settings["num_q"], model_settings["num_k"], model_settings["num_v"], model_settings["num_hiddens"], model_settings["dropout"]
        num_heads = model_settings["num_heads"] if output_structure == None else 1
        self.f_out, self.num_heads, self.output_structure = num_q//num_heads, num_heads, output_structure
        
        # linear mapping
        self.q_lin = nn.Linear(num_hiddens, num_q, bias = True)
        self.k_lin = nn.Linear(num_hiddens, num_k, bias = True)
        self.v_lin = nn.Linear(num_hiddens, num_v, bias = True)
        
        # nonlinear
        self.q_ff = FeedForward(f_in = num_q//num_heads, f_out = self.f_out, dropout = dropout)
        self.k_ff = FeedForward(f_in = num_k//num_heads, f_out = self.f_out, dropout = dropout)
        self.v_ff = FeedForward(f_in = num_v//num_heads, f_out = self.f_out, dropout = dropout)
        if output_structure == None:
            self.out = nn.Linear(self.f_out, self.f_out, bias = True)
        else:
            self.out = nn.Linear(self.f_out, 1, bias = True)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask = None):
        ## input size:  (batch, num_hidden, seq_len)
        # q = k = v = only_x
        ## self-attention
        # q = k = v = x_
        ## encoder output
        # q = y_t
        # k = v = e_output
        ## decoder output
        # q = k = v = torch.cat([x[:, :, 0].unsqueeze(-1), t.transpose(-1, -2)], axis = -1)
        # q = k = v = x
        # mask = d_m
        
        Q = self.q_ff(transpose_qkv(self.q_lin(q), self.num_heads))
        K = self.k_ff(transpose_qkv(self.k_lin(k), self.num_heads))
        V = self.v_ff(transpose_qkv(self.v_lin(v), self.num_heads))
        output = _attention(Q, K, V, self.f_out, mask, self.drop, self.output_structure)
        output = self.out(output)
            
        return transpose_output(output, self.num_heads)
        # output size: (batch * num_heads, seq_len, num_hidden / num_heads)
