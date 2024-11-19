# 1. The shape of the tensor passed across all modules are kept as (batch, d, seq_len)
# 2. Each module's (except for attention) output is normalized, so no need to normalize input

import torch, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Encoder import TransformerEncoder
from Decoder import TransformerDecoder
from SAND import SAND
from ModelUtils import FeedForward, compute_loss

# class Object(object):
#     pass
# self = Object()
# self = model

class Transformer(nn.Module):
    def __init__(self, model_settings):
        super().__init__()
        torch.manual_seed(321)
        model_settings["num_q"] = model_settings["num_k"] = model_settings["num_v"] = model_settings["num_hiddens"]
        self.model_settings = model_settings
        self.dataloader_settings = model_settings["dataloader_settings"]
        self.encoder = TransformerEncoder(model_settings)
        self.decoder = TransformerDecoder(model_settings)
        self.linear = nn.Linear(model_settings["num_hiddens"], 1)
        self.sigmoid = nn.Sigmoid()
        self.alpha_org, self.beta_org = model_settings["max_X"] - model_settings["min_X"], model_settings["min_X"]
        
        ## settings for SAND
        model_settings2 = model_settings.copy()
        model_settings2["num_hiddens"] = model_settings["f_in"][0]
        model_settings2["dropout"] = 0.05
        self.SAND = SAND(model_settings2)
    
    def forward(self, x, y_t, e_m_mh, d_m_mh, d_m, iteration = 0, TAs_position = None, isTraining = True):
        # x, y_t, e_m_mh, d_m_mh, d_m = x, d_T_full, e_m_mh, None, None
        # x, y_t, e_m_mh, d_m_mh, d_m = e_X, d_T, e_m_mh, d_m_mh, d_m  # for testing
        e_output = self.encoder(x, e_m_mh)
        d_output = self.decoder(y_t, e_output, e_m_mh, d_m_mh)
        org = self.linear(d_output) ## output from a vanilla Transformer
        org_detach = org.detach().clone()[:, :, 0].unsqueeze(-1)
        smooth = self.SAND(org_detach, y_t, d_m, iteration, TAs_position, isTraining) ## output from SAND
        return [smooth, self.sigmoid(org) * self.alpha_org + self.beta_org]
    
    def StartTraining(self, dataLoader, optimizer = None, save_model_every = 100, start_from_k = 0, verbose = False):
        if optimizer is None:
            optimizer = Adam(self.parameters(), lr = self.model_settings["lr"])
        
        if start_from_k > 0:
            ## resume training
            # checkpoint_tmp = torch.load("../Checkpoints/Tmp" + ("/IID" if iidt else "/NonIID") + ("/RealData" if real else "/Simulation") + data_name + output_structure_folder + sparsity_error_folder + "/ckpts_l2w_1900.pth", map_location = device)
            # model.load_state_dict(checkpoint_tmp["state_dict"])
            # for parameter in model.parameters():
            #     parameter.requires_grad = True
            # model.train()
            # optimizer.load_state_dict(checkpoint_tmp['optimizer'])
            pass
        
        model_settings = self.model_settings
        dataloader_settings = self.dataloader_settings
        sparsity_error_folder = model_settings["sparsity_error_folder"]
        device = model_settings["device"]
        data_name = dataloader_settings["data_name"]
        repeat_obj = torch.tensor([self.model_settings["num_heads"]] * self.dataloader_settings["batch_size"]).to(device)
        
        loss_train_history = []
        loss_valid_history = []
        min_valid_loss = sys.maxsize
        for k in range(start_from_k, model_settings["max_epoch"]):
            if k % save_model_every == 0 or k == 100:
                checkpoint = {"model": self.model_settings,
                              "state_dict": self.state_dict(), 
                              "optimizer" : optimizer.state_dict()}
                torch.save(checkpoint, "../Checkpoints/" + data_name + sparsity_error_folder + "/ckpts_" + str(k) + ".pth")
            
            loss_train = []
            loss_valid = []
            dataLoader.shuffle()
            self = self.train()
            # set model training state
            for i, (emb_mask, e_m, d_m, x, y_t, y) in enumerate(dataLoader.get_train_batch()):
                # stop
                # e_m: mask for encoder, randomly mask some points; d_m: mask for decoder, only mask unobserved points
                # x: [B, 2, obs]: x and t, y_t: t, y: x
                optimizer.zero_grad()
                e_m_mh = torch.repeat_interleave(e_m, model_settings["num_heads"], dim = 0)
                d_m_mh = torch.repeat_interleave(d_m, model_settings["num_heads"], dim = 0)
                out = self.forward(x, y_t, e_m_mh, d_m_mh, d_m, iteration = k)
                loss = compute_loss(out, y, d_m, k, isTraining = True)
                loss_train.append(loss.item())
                
                # update parameters using backpropagation
                loss.backward()
                optimizer.step()
            avg_loss_train = np.mean(loss_train)
            loss_train_history.append(avg_loss_train)
            # scheduler.step(avg_loss_train)
            
            # model evaluation mode
            with torch.no_grad():
                self = self.eval()
                for emb_mask, e_m, d_m, x, y_t, y in dataLoader.get_valid_batch():
                    # stop
                    e_m_mh = torch.repeat_interleave(e_m, model_settings["num_heads"], dim = 0)
                    d_m_mh = torch.repeat_interleave(d_m, model_settings["num_heads"], dim = 0)
                    valid_y = self.forward(x, y_t, e_m_mh, d_m_mh, d_m, isTraining = False)
                    loss_valid_tmp = compute_loss(valid_y, y, d_m, k, isTraining = False)
                    loss_valid.append(loss_valid_tmp.item())
        
            if np.mean(loss_valid) < min_valid_loss:
                print("The best model is " + str(k) + ".", flush = True)
                checkpoint_best = {"model": self.model_settings,
                                   "state_dict": self.state_dict(), 
                                   "optimizer" : optimizer.state_dict()}
                torch.save(checkpoint_best, "../Checkpoints/" + data_name + sparsity_error_folder + "/best_ckpts.pth")
                min_valid_loss = np.mean(loss_valid)
        
            loss_valid_history.append(np.mean(loss_valid))
            if (k < 20 or k % 50 == 0) and verbose:
                print("epoch:", k, "training loss = ", loss_train_history[-1], "validation loss = ", loss_valid_history[-1], flush = True)
    
def GetImputation(X_obs, T_obs, data_name = "HighDim_E", sparsity = "dense", error = False):
    # X_obs = X_test
    # T_obs = T_test
    
    assert X_obs.shape[0] == T_obs.shape[0]
    assert X_obs.shape[1] - 1 == T_obs.shape[1]
    
    # Load checkpoint
    device = torch.device("cpu")
    sparsity_error_folder = "/" + ((sparsity + "/w_error" if error else sparsity + "/wo_error") if data_name != "UK" else sparsity) if data_name != "Framingham" else ""
    checkpoint = torch.load("../Checkpoints/" + data_name + sparsity_error_folder + "/best_ckpts.pth")
    model = Transformer(checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.model_settings["device"] = device
    
    for parameter in model.parameters():
        parameter.requires_grad = False
            
    # send model to CPU
    model = model.to(device)
    model = model.eval()
    num_heads = model.model_settings["num_heads"]
    
    L = model.model_settings["tptsTraining"]
    t_true = np.linspace(0, 1, L)
    d = model.model_settings["f_in"][0] - 2
    d_T = np.zeros([1, d + 1, L])
    
    d_T[:, 0, :] = t_true # before multiply by 100
    for i in range(int(d/2)):
        d_T[0, 2*i+1, :] = np.sin(10 ** (- 4*(i + 1)/d) * t_true * (L - 1))
        d_T[0, 2*i+2, :] = np.cos(10 ** (- 4*(i + 1)/d) * t_true * (L - 1))
            
    d_T = torch.Tensor(d_T).to(device)
    d_m_mh = torch.Tensor([[1] * L] * num_heads).to(device)
    d_m = torch.Tensor([[1] * L]).to(device)
    
    org_pred = []
    smooth_pred = []
    prob_list = torch.Tensor(np.array([np.exp((1 - torch.abs(torch.Tensor(range(-L, L + 1))/L)) * 50).numpy()]))
    prob_list = prob_list/prob_list.sum(1, keepdim = True) + (2e-16)
    prob_list = prob_list/prob_list.sum(1, keepdim = True)
    
    M_obs = np.array(X_obs.iloc[:, 0], dtype = int)
    X_obs = X_obs.iloc[:, 1:]
    n, m = X_obs.shape
    # i = 0
    for i in range(X_obs.shape[0]):
        # stop
        src_mask = [1] * m
        src_mask[M_obs[i]:] = [0] * (m - M_obs[i])
        x = X_obs.iloc[i, :]
        t = np.array(T_obs.iloc[i, :])
        
        e_XT = np.zeros([1, d + 1, len(t)])
        e_XT[:, 0, :] = t # before multiply by 100
        for j in range(int(d/2)):
            e_XT[0, 2*j+1, :] = np.sin(10 ** (- 4*(j + 1)/d) * t * (L - 1))
            e_XT[0, 2*j+2, :] = np.cos(10 ** (- 4*(j + 1)/d) * t * (L - 1))
            
        e_X = torch.Tensor(np.concatenate((np.array([[x]]), e_XT), axis = 1))
        e_m = torch.Tensor(src_mask).reshape(1, -1)
        e_m_mh = torch.repeat_interleave(e_m, num_heads, dim = 0).to(device)
        
        TAs_position = np.array(e_X[0, 1, :] * (L - 1), dtype = int)
        TAs_position = torch.Tensor(TAs_position[np.diff(np.concatenate(([-1], TAs_position))) > 0]).unsqueeze(-1).to(device)
        [smooth, org] = model.forward(e_X.to(device), d_T, e_m_mh, d_m_mh, d_m, TAs_position = TAs_position)
        weight = torch.zeros(smooth.size()[0], smooth.size()[1], 1)
        for j, pos in enumerate(TAs_position.squeeze(-1).long()):
            weight[j, :, 0] = prob_list[0][(L - pos + 1):(2 * L - pos + 1)]
        
        weight = (weight/weight.sum(0)).to(device)
        smooth_prob = torch.sum(smooth * weight, dim = 0)
        smooth_pred.append(smooth_prob.cpu().numpy())
        org_pred.append(org.squeeze(0).cpu().numpy())
    return [np.array(smooth_pred).squeeze(-1), np.array(org_pred).squeeze(-1)]
