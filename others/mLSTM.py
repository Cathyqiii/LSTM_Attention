#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/5/15 22:19
@desc: test model
"""
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class mLSTM(torch.nn.Module):
    def __init__(self, args):
        super(mLSTM, self).__init__()
        self.enc_seq_len = args.interval*args.lag
        self.pre_len = args.window*args.horizon
        self.lstm_enc_in = 1
        self.lstm_out_dim = 128
        self.encode = nn.LSTM(self.lstm_enc_in, self.lstm_out_dim, 1)
        self.decoder = nn.LSTM(self.lstm_out_dim, self.lstm_out_dim, 1)

        self.fc = nn.Linear(self.lstm_out_dim, self.pre_len)
        self.activate = nn.ReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        sample_num = x.shape[0]
        # x = x.unsqueeze(2)
        h0 = torch.randn(1, sample_num, self.lstm_out_dim).to(x.device)
        c0 = torch.randn(1, sample_num, self.lstm_out_dim).to(x.device)

        seq_last = x[:,-1,:].unsqueeze(1)
        x = x - seq_last

        x = x.transpose(0,1)
        out, (hn, cn) = self.encode(x, (h0, c0))
        out, (hn, cn) = self.decoder(out, (hn, cn))

        out = self.fc(out[-1]) #共享一个fc
        out = self.drop(out)
        out.unsqueeze_(2)

        out = out + seq_last

        return out