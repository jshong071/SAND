import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import os

# class Object(object):
#     pass
# self = Object()
# random_seed = 321

def getData(data_name, sparsity = "dense", error = False):
    ## dataset is created using R. Here we just use the data.
    sparsity_error_folder = "/" + ((sparsity + "/w_error" if error else sparsity + "/wo_error") if data_name != "UK" else sparsity) if data_name != "Framingham" else ""
    folder_path = "../Data/" + data_name + sparsity_error_folder
    
    if os.path.exists(folder_path + "/X_obs.csv") and os.path.exists(folder_path + "/T_obs.csv"):
        X = pd.read_csv(folder_path + "/X_obs.csv", header = None)
        T = pd.read_csv(folder_path + "/T_obs.csv", header = None)
    else:
        raise Exception("X_obs.csv or T_obs.csv is not found in " + os.path.abspath(folder_path) + ".\n Please use R to simulate the dataset.")
    
    return X, T, sparsity_error_folder

class DataLoader:
    def __init__(self, X, T, dataloader_settings, random_seed = 321):
        # train/valid/test split
        np.random.seed(random_seed)
        self.device = dataloader_settings["device"]
        d, split = dataloader_settings["d"], dataloader_settings["split"]
        num_obs = np.array(X.iloc[:, 0], dtype = int)
        X = X.iloc[:, 1:]
        self.n, self.m = X.shape
        
        # preprocess on X
        X.replace(np.nan, 0, inplace = True)
        
        # preprocess on T
        T.replace(np.nan, 0, inplace = True)
        self.tptsTesting = len(np.unique(T))
        
        full_prob_list = np.zeros((self.n, max(num_obs) * 2 + 1))
        for i in range(self.n):
            full_prob_list[i, :num_obs[i] * 2 + 1] = norm.pdf(x = np.array(range(-num_obs[i], num_obs[i] + 1)), loc = 0, scale = (T.iloc[i, num_obs[i] - 1] - T.iloc[i, 0]) * 15)
        
        encT = np.zeros([self.n, d, self.m])
        for i in range(int(d/2)):
            encT[:, 2 * i, :] = np.sin(10 ** (- 4*(i + 1)/d) * T * (self.tptsTesting - 1))
            encT[:, 2 * i + 1, :] = np.cos(10 ** (- 4*(i + 1)/d) * T * (self.tptsTesting - 1))
        
        self.batch_size = dataloader_settings["batch_size"]
        self.train_n = round(self.n * split[0] / sum(split))
        self.valid_n = round(self.n * split[1] / sum(split))
        self.test_n = self.n - self.train_n - self.valid_n
        
        self.train_X = torch.Tensor(np.array(X.iloc[:self.train_n, :])).to(self.device).unsqueeze(1) # (batchsize, 1, numpts)
        self.train_T = torch.Tensor(np.array(T.iloc[:self.train_n, :])).to(self.device).unsqueeze(1) # (batchsize, 1, numpts)
        self.train_encT = torch.Tensor(encT[:self.train_n, :, :]).to(self.device) # (batchsize, d, numpts)
        self.train_O = num_obs[:self.train_n]
        self.train_prob_list = torch.Tensor(full_prob_list[:self.train_n, :]).to(self.device) # (batchsize, numpts*2+1)
        
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
        
        self.max_X, self.min_X = torch.max(self.train_X).to(self.device), torch.min(self.train_X).to(self.device)
        
        # preprocess on full T
        full_T = np.linspace(0, 1, len(np.unique(T)))
        enc_full_T = np.zeros([1, d + 1, len(full_T)])
        enc_full_T[:, 0, :] = full_T
        for i in range(int(d/2)):
            enc_full_T[:, 2 * i + 1, :] = np.sin(10 ** (- 4*(i + 1)/d) * full_T * (self.tptsTesting - 1))
            enc_full_T[:, 2 * i + 2, :] = np.cos(10 ** (- 4*(i + 1)/d) * full_T * (self.tptsTesting - 1))
            
        self.tptsTraining = full_T.shape[-1]
        self.full_T = torch.Tensor(enc_full_T).to(self.device) # (1, d + 1, allpts)

    def shuffle(self): # For each epoch, we shuffle the order of inputs.
        # training dataset
        new_order = np.arange(self.train_X.shape[0])
        np.random.shuffle(new_order)
        self.train_X = self.train_X[new_order]
        self.train_T = self.train_T[new_order]
        self.train_encT = self.train_encT[new_order]
        self.train_O = self.train_O[new_order]
        self.train_prob_list = self.train_prob_list[new_order]
        
    def _batch_generator(self, X, T, encT, O, n, prob_list): # n: sample size, B: batch size
        # i, X, T, encT, O, n, prob_list = 0, self.train_X, self.train_T, self.train_encT, self.train_O, self.train_n, self.train_prob_list
        def generator_func():
            N = round(np.ceil(n/self.batch_size))
            treated_as_noniid = (np.random.randint(low = 0, high = 2, size = N) == 1)
            y_t = torch.repeat_interleave(self.full_T, self.batch_size, axis = 0)
                
            for i in range(N):
                x = X[(i * self.batch_size):((i + 1)*self.batch_size)]
                t = T[(i * self.batch_size):((i + 1)*self.batch_size)]
                enct = encT[(i * self.batch_size):((i + 1)*self.batch_size)]
                obs = O[(i * self.batch_size):((i + 1)*self.batch_size)]
                prob_list_now = prob_list[(i * self.batch_size):((i + 1)*self.batch_size)]
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
                # src: [x, t, encoded t]; y_t: [t, encoded t], for decoder, y: only x, for prediction
                # [d_mask[j, index_mat[j, range(obs[j])]] for j in range(self.batch_size)]
        return generator_func()

    def get_train_batch(self):
        return self._batch_generator(self.train_X, self.train_T, self.train_encT, self.train_O, self.train_n, self.train_prob_list)

    def get_valid_batch(self):
        return self._batch_generator(self.valid_X, self.valid_T, self.valid_encT, self.valid_O, self.valid_n, self.valid_prob_list)

    def get_test_batch(self):
        return self._batch_generator(self.test_X, self.test_T, self.test_encT, self.test_O, self.test_n, self.test_prob_list)

