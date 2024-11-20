import math
import pandas as pd
import torch
from torch import nn
from Attention import MultiHeadAttention
from ModelUtils import Norm, FeedForward

# class Object(object):
#     pass
# self = Object()

class EncoderBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, model_settings):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(model_settings)
        self.ff = FeedForward(model_settings["num_hiddens"], model_settings["num_hiddens"], dropout = model_settings["dropout"])
        self.norm1 = Norm(model_settings["num_hiddens"], axis = -1)
        self.norm2 = Norm(model_settings["num_hiddens"], axis = -1)
        self.drop1 = nn.Dropout(model_settings["dropout"])
        self.drop2 = nn.Dropout(model_settings["dropout"])

    def forward(self, x_, mask):
        x_ = self.norm1(x_ + self.drop1(self.attention(x_, x_, x_, mask)))
        x_ = x_ + self.drop2(self.ff(x_))
        return self.norm2(x_)

class TransformerEncoder(nn.Module):
    """Transformer encoder"""
    def __init__(self, model_settings):
        super(TransformerEncoder, self).__init__()
        embedding_model_settings = model_settings.copy()
        embedding_model_settings["num_hiddens"] = 1
        self.embedding_atte = MultiHeadAttention(embedding_model_settings)
        self.embedding_drop = nn.Dropout(model_settings["dropout"])
        self.embedding_norm = Norm(model_settings["num_hiddens"], axis = -1)
        
        self.broadcast = nn.Linear(model_settings["f_in"][0], model_settings["num_hiddens"], bias = True)
        self.ff = FeedForward(model_settings["num_hiddens"], model_settings["num_hiddens"], dropout = model_settings["dropout"])
        self.norm1 = Norm(model_settings["num_hiddens"], axis = -1)
        self.norm2 = Norm(model_settings["num_hiddens"], axis = -1)
        self.blks = nn.Sequential()
        for i in range(model_settings["num_layers"][0]):
            self.blks.add_module("block" + str(i), EncoderBlock(model_settings))

    def forward(self, x, mask):
        # input x size    - (B, 2, seq_len)
        x_ = self.ff(self.norm1(self.broadcast(x.transpose(-1, -2))))
        # output x size   - (B, seq_len, d)
        
        for i, blk in enumerate(self.blks):
            # mask size - (B, seq_len)
            x_ = blk(x_, mask)
        return self.norm2(x_)

