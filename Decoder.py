import math
import pandas as pd
import torch
from torch import nn
from Attention import MultiHeadAttention
from ModelUtils import Norm, FeedForward

# class Object(object):
#     pass
# self = Object()

class DecoderBlock(nn.Module):
    """Transformer decoder block"""
    def __init__(self, model_settings): #num_q, num_k, num_v, num_hiddens, num_heads, dropout):
        super(DecoderBlock, self).__init__()
        self.attention1 = MultiHeadAttention(model_settings)
        self.attention2 = MultiHeadAttention(model_settings)
        self.ff = FeedForward(model_settings["num_hiddens"], model_settings["num_hiddens"], dropout = model_settings["dropout"])
        self.norm1 = Norm(model_settings["num_hiddens"], axis = -1)
        self.norm2 = Norm(model_settings["num_hiddens"], axis = -1)
        self.norm3 = Norm(model_settings["num_hiddens"], axis = -1)
        self.drop1 = nn.Dropout(model_settings["dropout"])
        self.drop2 = nn.Dropout(model_settings["dropout"])

    def forward(self, y_t2, e_output, e_mask, d_mask):
        y_t2 = y_t2 + self.drop1(self.attention1(y_t2, y_t2, y_t2, d_mask))
        y_t2 = self.norm1(y_t2)
        
        y_t2 = y_t2 + self.drop2(self.attention2(y_t2, e_output, e_output, e_mask))
        y_t2 = self.norm2(y_t2)
        
        y_t2 = y_t2 + self.ff(y_t2)
        return self.norm3(y_t2)

class TransformerDecoder(nn.Module):
    """Transformer decoder"""
    def __init__(self, model_settings):
        super(TransformerDecoder, self).__init__()
        self.broadcast = nn.Linear(model_settings["f_in"][1], model_settings["num_hiddens"], bias = True)
        self.ff = FeedForward(model_settings["num_hiddens"], model_settings["num_hiddens"], dropout = model_settings["dropout"])
        self.norm1 = Norm(model_settings["num_hiddens"], axis = -1)
        self.norm2 = Norm(model_settings["num_hiddens"], axis = -1)
        self.blks = nn.Sequential()
        for i in range(model_settings["num_layers"][1]):
            self.blks.add_module("block" + str(i), DecoderBlock(model_settings))
        
    def forward(self, y_t, e_output, e_mask, d_mask):
        # input y_t size:      (batch_num, seq_len, 1)
        # input e_output size: (batch_num, num_hiddens, seq_len)
        # input e_mask size:   (batch_num, seq_len)
        # input d_mask size:   (batch_num, seq_len)
        # y_t, e_output, e_mask, d_mask = y_t, e_output, e_m_mh, d_m_mh
        
        y_t2 = self.ff(self.norm1(self.broadcast(y_t.transpose(-1, -2))))
        for i, blk in enumerate(self.blks):
            y_t2 = blk(y_t2, e_output, e_mask, d_mask)
        return self.norm2(y_t2)


