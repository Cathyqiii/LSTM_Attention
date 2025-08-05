#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/5/15 22:19
@desc: test models
"""
import torch
import torch.nn as nn
# import torch.nn.functional as F
import warnings
# import random
# from others.lstm_tf import Encoder
warnings.filterwarnings("ignore")


# Attention机制实现
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states):
        attn_weights = torch.tanh(self.attn(hidden_states))
        attn_weights = self.context(attn_weights).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights.unsqueeze(2) * hidden_states, dim=1)
        return context_vector, attn_weights

'''
# LSTM模型实现
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        context_vector, attn_weights = self.attention(out)
        out = self.fc(context_vector)
        return out, attn_weights
'''

# LSTM模型实现
class Attn_LSTM(nn.Module):
    def __init__(self, args):
        super(Attn_LSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.d_model
        self.n_layers = args.e_layers
        self.lstm = nn.LSTM(args.enc_in, self.hidden_dim, self.n_layers)
        self.attention = Attention(self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, args.pred_len)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        h_0 = torch.zeros(self.n_layers, x.size(1), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.n_layers, x.size(1), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        context_vector, attn_weights = self.attention(out)
        out = self.fc(context_vector)
        if self.args.output_attention:
            return out.unsqueeze(2), attn_weights
        else:
            return out.unsqueeze(2)

