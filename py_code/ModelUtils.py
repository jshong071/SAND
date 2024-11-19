import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# class Object(object):
#     pass
# self = Object()

def compute_loss(y_pred, y, mask, iteration, isTraining = True):
    # y_pred, mask = out, d_m
    # y_pred, mask = valid_y, d_m
    # dim - (B, J)
    # mean square error
    [smooth, original] = y_pred
    
    smoothMSE = torch.pow(torch.sum((smooth[:, :, 0] - y) ** 2 * mask) / torch.sum(mask), 1/2)
    smoothMSE_train = smoothMSE.clone()
    
    originalMSE = torch.pow(torch.sum((original[:, :, 0] - y) ** 2 * mask) / torch.sum(mask), 1/2)
    originalMSE_train = originalMSE.clone()
    if isTraining:
        return originalMSE_train * 0.75 + smoothMSE_train * 0.25
    else:
        return smoothMSE

class Norm(nn.Module):
    def __init__(self, d, axis = -2, eps = 1e-6):
        super().__init__()
        self.d = d
        self.axis = axis
        # create two learnable parameters to calibrate normalisation
        if axis == -2:
            self.alpha = nn.Parameter(torch.randn((d, 1)))
            self.bias = nn.Parameter(torch.randn((d, 1)))
        else:
            self.alpha = nn.Parameter(torch.ones(d))
            self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        avg = x.mean(dim=self.axis, keepdim = True) if self.d > 1 else 0
        std = x.std(dim=self.axis, keepdim = True) + self.eps if self.d > 1 else 1
        norm = self.alpha * (x - avg) / std + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, f_in, f_out, hidden = 64, dropout = 0.1):
        super().__init__()
        self.hidden = hidden
        self.lin1 = nn.Linear(f_in, self.hidden)
        self.lin2 = nn.Linear(self.hidden, self.hidden)
        self.lin3 = nn.Linear(self.hidden, f_out)
        self.norm1 = Norm(self.hidden, axis = -1)
        self.norm2 = Norm(self.hidden, axis = -1)
        self.norm3 = Norm(f_out, axis = -1)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, useReLU = True):
        if useReLU:
            x_ = F.relu(self.lin1(x))
            x_ = x_ + F.relu(self.lin2(x_)) ## residual network
            x_ = self.norm1(x_)
            x_ = x_ + F.relu(self.lin2(self.drop1(x_)))
            x_ = self.norm2(x_)
            x_ = F.relu(self.lin3(self.drop2(x_)))
        else:
            x_ = F.gelu(self.lin1(x))
            x_ = x_ + F.gelu(self.lin2(x_))
            x_ = self.norm1(x_)
            x_ = x_ + F.gelu(self.lin2(self.drop1(x_)))
            x_ = self.norm2(x_)
            x_ = F.gelu(self.lin3(self.drop2(x_)))
        
        return self.norm3(x_)

