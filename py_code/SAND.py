import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from Attention import MultiHeadAttention
# org_detach = org.detach().clone()

# self = self.outblock
class SAND(nn.Module):
    """SAND"""
    def __init__(self, model_settings):
        super(SAND, self).__init__()
        # model_settings_tmp = model_settings
        # model_settings = model_settings2
        self.model_settings = model_settings
        self.outblks = nn.Sequential()
        self.sigmoid = nn.Sigmoid()
        self.alpha, self.beta = 1 * (model_settings["max_X"] - model_settings["min_X"]), model_settings["min_X"]
        self.outblks.add_module("BlockSAND", MultiHeadAttention(model_settings, "SAND"))
            
        n_timepts = model_settings["tptsTraining"]
        self.timediff = 1/(n_timepts - 1)
        self.index_subs = np.repeat(range(model_settings["batch_size"]), n_timepts)
        self.range_pts = torch.arange(n_timepts, device = model_settings["device"]).unsqueeze(0)
        weight_rl = np.array(range(n_timepts))/(n_timepts - 1)
        weight_lr = 1 - weight_rl
        self.weight_lr_SL = torch.Tensor(weight_lr).to(model_settings["device"])
        self.weight_rl_SL = torch.Tensor(weight_rl).to(model_settings["device"])
        weight_lr_tmp = torch.Tensor(np.exp(weight_lr * 20))
        self.weight_lr = (weight_lr_tmp/weight_lr_tmp.sum() + (2e-16)).to(model_settings["device"])    
        # model_settings = model_settings_tmp

    def forward(self, org_detach, y_t, d_m, iteration = 0, TAs_position = None, isTraining = True):
        # outblock_input - (B, seq_len, num_hiddens)
        # d_m            - (B, seq_len)
        device = self.model_settings["device"] if TAs_position is None else torch.device("cpu")
        smooth = torch.zeros(org_detach.size(), device = device)
        for i, blk in enumerate(self.outblks):
            outblock_input = torch.cat([org_detach[:, :, i].unsqueeze(-1), y_t.transpose(-1, -2)], axis = -1)
            smooth[:, :, i] = blk(outblock_input, outblock_input, outblock_input, mask = d_m).squeeze(-1)
        
        [n_subs, n_timepts, _] = org_detach.size()
        if TAs_position == None:
            # iteration = 0; isTraining = True
            TAs_num = np.random.randint(low = 2, high = max(15 - iteration/300, 3)) if isTraining else 2
            TAs_position, _ = torch.sort(torch.multinomial(torch.ones(n_subs, int(n_timepts - 2)), int(TAs_num)) + 1)
            TAs_position = TAs_position.to(device)
            index_subs = self.index_subs if n_subs == self.model_settings["batch_size"] else np.repeat(range(n_subs), n_timepts)
        else:
            assert n_subs == 1
            n_subs = len(TAs_position)
            index_subs = np.repeat(0, n_timepts * n_subs)
        
        increments_lr = torch.cumsum(self.timediff * smooth, dim = 1) # increment compared to the first point
        TAs_position_lr = torch.cat((torch.zeros(n_subs, 1, device = device), TAs_position), dim = 1)
        repeats = torch.diff(torch.cat((TAs_position_lr, torch.full((n_subs, 1), n_timepts, device = device)), dim = 1), dim = 1).reshape(-1).long()
        index_timepts_lr = torch.repeat_interleave(TAs_position_lr.reshape(-1), repeats).long()
        smooth_lr = (org_detach[index_subs, index_timepts_lr, :] - increments_lr[index_subs, index_timepts_lr, :]).reshape(n_subs, n_timepts, -1) + increments_lr
        
        increments_rl = torch.cumsum(self.timediff * smooth[:, range(-1, -(n_timepts+1), -1), :], dim = 1)[:, range(-1, -(n_timepts+1), -1), :] # increment compared to the last point
        TAs_position_rl = torch.cat((TAs_position, torch.full((n_subs, 1), n_timepts - 1, device = device)), dim = 1)
        repeats = torch.diff(torch.cat((torch.full((n_subs, 1), -1, device = device), TAs_position_rl), dim = 1), dim = 1).reshape(-1).long()
        index_timepts_rl = torch.repeat_interleave(TAs_position_rl.reshape(-1), repeats).long()
        smooth_rl = (org_detach[index_subs, index_timepts_rl, :] + increments_rl[index_subs, index_timepts_rl, :]).reshape(n_subs, n_timepts, -1) - increments_rl
        
        TAs_position_lr = index_timepts_lr.reshape(n_subs, -1)
        TAs_position_rl = index_timepts_rl.reshape(n_subs, -1)
        weight_lr = self.weight_lr[self.range_pts - TAs_position_lr]
        weight_rl = self.weight_lr[TAs_position_rl - self.range_pts]
        weight_sum = weight_lr + weight_rl
        
        smooth_out = self.sigmoid(smooth_lr * (weight_lr/weight_sum).unsqueeze(-1) + smooth_rl * (weight_rl/weight_sum).unsqueeze(-1))
        return smooth_out * self.alpha + self.beta
