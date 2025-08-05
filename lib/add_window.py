#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/8/25 20:42
@desc:
"""
import torch
import pickle


def Add_Window_Horizon(data, args):
    '''
    :param data: [T N D]
    :param window: the length of historical sequence
    :param horizon:
    :param single:
    :return:
    '''
    #得到x,y
    X = []
    Y = []
    p, lag, horizon = args.interval, args.lag, args.horizon
    seq_len = p*lag
    pred_len = p*horizon
    dlen = len(data)
    for index in range(0, dlen-seq_len-pred_len, 5):
        s_begin = index
        s_end = s_begin + seq_len
        r_begin = s_end
        r_end = r_begin + pred_len

        X.append(data[s_begin:s_end])
        Y.append(data[r_begin:r_end])

    return torch.stack(X,dim=0), torch.stack(Y, dim=0)

def Add_Window_Horizon_stamp(data, dTime, args):
    '''
    :param data: [T N D]
    :param window: the length of historical sequence
    :param horizon:
    :param single:
    :return:
    '''
    #得到x,y
    X = []
    Y = []
    X_stamp = []
    Y_stamp = []

    p, lag, horizon, label_len = args.interval, args.lag, args.horizon, args.label_len

    seq_len = p*lag
    pred_len = p*horizon
    dlen = len(data)
    for index in range(0, dlen-seq_len-pred_len, 5):
        s_begin = index
        s_end = s_begin + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + pred_len + label_len

        X.append(data[s_begin:s_end])
        Y.append(data[r_begin:r_end])

        X_stamp.append(dTime[s_begin:s_end])
        Y_stamp.append(dTime[r_begin:r_end])

    return torch.stack(X,dim=0), torch.stack(Y, dim=0), torch.stack(X_stamp,dim=0), torch.stack(Y_stamp, dim=0)

def Add_Window_Horizon_DF(data, p, lag, horizon):
    '''
    :param data: [T N D]
    :param window: the length of historical sequence
    :param horizon:
    :param single:
    :return:
    '''
    #得到x,y
    X = []
    Y = []
    seq_len = p*lag
    pred_len = p*horizon
    dlen = len(data)
    for index in range(0, dlen-seq_len-pred_len, 5):
        s_begin = index
        s_end = s_begin + seq_len
        r_begin = s_end
        r_end = r_begin + pred_len

        X.append(data[s_begin:s_end])
        Y.append(data[r_begin:r_end])

    return X, Y

if __name__ == '__main__':

    #data = load_st_dataset('Wind')
    with open('./data/Wind/wind.file', 'rb') as fo:
        data = pickle.load(fo)
    X_enc_set, X_dec_set, Y_power_set = Add_Window_Horizon(data=data, window=192, horizon=96)
    print(X_enc_set.shape, X_dec_set.shape, Y_power_set.shape)
#  PEMS08: X [17852 3 170 1] Y [17852 2 170 1]