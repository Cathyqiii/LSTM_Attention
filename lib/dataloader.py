#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/9/12 20:47
@desc:
"""
import torch
import torch.utils.data
import pandas as pd
import numpy as np
from lib.informers.timefeatures import time_features
from lib.add_window import Add_Window_Horizon, Add_Window_Horizon_stamp
from lib.load_dataset import load_st_dataset
from lib.normalization import NScalar, MinMax01Scaler, MinMax11Scaler, StandardScalar, ColumnMinMaxScaler
import random
from TSlib.data_provider.data_factory import data_provider
from TSlib.data_provider.data_loader import Dataset_M4
from TSlib.data_provider.m4 import M4Meta
from torch.utils.data import DataLoader

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum.values, maximum.values)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScalar(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScalar()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        # column min max, to be depressed
        # note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler


def split_data_by_ratio_stamp(data, dTime, args):
    train_data = []
    test_data = []
    val_data = []
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    data_len = len(data)

    timeenc = args.timeenc
    freq = args.freq
    dTime['date'] = pd.to_datetime(dTime.date)
    data_stamp = time_features(dTime, timeenc=timeenc, freq=freq)
    data_stamp = torch.tensor(data_stamp)


    if data_len > 12 * 30 * 24 + 8 * 30 * 24:
            #border1s = [4 * 30 * 24,     12 * 30 * 24,               12 * 30 * 24 + 6 * 30 * 24]
            #border2s = [12 * 30 * 24,    12 * 30 * 24 + 2 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            border1s = [0,            12 * 30 * 24,               12 * 30 * 24 + 4 * 30 * 24]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

            train_data = data[border1s[0]:border2s[0]]
            test_data = data[border1s[1]:border2s[1]]
            val_data = data[border1s[2]:border2s[2]]

            train_data_mark = data_stamp[border1s[0]:border2s[0]]
            test_data_mark = data_stamp[border1s[1]:border2s[1]]
            val_data_mark = data_stamp[border1s[2]:border2s[2]]

    else:
            test_data = data[-int(data_len*test_ratio):]
            val_data = data[-int(data_len*(test_ratio + val_ratio)): -int(data_len*test_ratio)]
            train_data = data[:-int(data_len*(test_ratio+val_ratio))]

            test_data_mark = data_stamp[-int(data_len*test_ratio):]
            val_data_mark = data_stamp[-int(data_len*(test_ratio + val_ratio)): -int(data_len*test_ratio)]
            train_data_mark = data_stamp[:-int(data_len*(test_ratio+val_ratio))]

    # test_data = data[-int(data_len*test_ratio):]
    # val_data = data[-int(data_len*(test_ratio + val_ratio)): -int(data_len*test_ratio)]
    # train_data = data[:-int(data_len*(test_ratio+val_ratio))]

    # test_data_mark = data_stamp[-int(data_len*test_ratio):]
    # val_data_mark = data_stamp[-int(data_len*(test_ratio + val_ratio)): -int(data_len*test_ratio)]
    # train_data_mark = data_stamp[:-int(data_len*(test_ratio+val_ratio))]

    return train_data, val_data, test_data, train_data_mark, test_data_mark, val_data_mark

def split_data_by_ratio(data, args):
    train_data = []
    test_data = []
    val_data = []
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    data_len = len(data)

    if data_len > 12 * 30 * 24 + 8 * 30 * 24:
            #border1s = [4 * 30 * 24,     12 * 30 * 24,               12 * 30 * 24 + 6 * 30 * 24]
            #border2s = [12 * 30 * 24,    12 * 30 * 24 + 2 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            border1s = [0,            12 * 30 * 24,               12 * 30 * 24 + 4 * 30 * 24]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

            train_data = data[border1s[0]:border2s[0]]
            test_data = data[border1s[1]:border2s[1]]
            val_data = data[border1s[2]:border2s[2]]
    else:
            test_data = data[-int(data_len*test_ratio):]
            val_data = data[-int(data_len*(test_ratio + val_ratio)): -int(data_len*test_ratio)]
            train_data = data[:-int(data_len*(test_ratio+val_ratio))]

    # test_data = data[-int(data_len * test_ratio):]
    # val_data = data[-int(data_len * (test_ratio + val_ratio)): -int(data_len * test_ratio)]
    # train_data = data[:-int(data_len * (test_ratio + val_ratio))]

    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    # cuda = True if torch.cuda.is_available() else False
    # TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # X = X.float()
    # Y = Y.float()
    # X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def data_loader_stamp_metr_pems(x_tra, y_tra, x_tra_stamp, y_tra_stamp, batch_size, shuffle=True, drop_last=True):
    n_total = x_tra.shape[0]
    t_id = random.sample(range(n_total), int(n_total * 0.01))
    x_tra = x_tra[t_id]
    y_tra = y_tra[t_id]
    x_tra_stamp = x_tra_stamp[t_id]
    y_tra_stamp = y_tra_stamp[t_id]

    X_and_stamp = []
    Y_and_stamp = []
    for i in range(x_tra.shape[2]):
        X_and_stamp.append(torch.cat((x_tra[:, :, i].unsqueeze(2), x_tra_stamp), dim=2))
        Y_and_stamp.append(torch.cat((y_tra[:, :, i].unsqueeze(2), y_tra_stamp), dim=2))
    X_and_stamp = torch.vstack(X_and_stamp)
    Y_and_stamp = torch.vstack(Y_and_stamp)

    data = torch.utils.data.TensorDataset(X_and_stamp, Y_and_stamp)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)

    return dataloader

def data_loader_stamp(X, Y, X_stamp, Y_stamp, batch_size, shuffle=True, drop_last=True):
    # cuda = True if torch.cuda.is_available() else False
    # TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # X = X.float()
    # Y = Y.float()
    # X_stamp = X_stamp.float()
    # Y_stamp = Y_stamp.float()
    #
    # X, Y ,X_stamp, Y_stamp = TensorFloat(X), TensorFloat(Y), TensorFloat(X_stamp), TensorFloat(Y_stamp)
    if len(X.shape) == 2:
        X = X.unsqueeze(2)
        Y = Y.unsqueeze(2)
    X_and_stamp = torch.cat((X, X_stamp), dim=2)
    Y_and_stamp = torch.cat((Y, Y_stamp), dim=2)

    data = torch.utils.data.TensorDataset(X_and_stamp, Y_and_stamp)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def delete_outlier(array_x, array_y):
    '''
    :param array_x: Samples*Lag*Dim
    :param array_y: Samples*Horizon
    :return: arrays without outliers
    '''
    index = np.argwhere(array_x == -99)
    index_ = np.unique(index[:, 0])
    x = np.delete(array_x, index_, axis=0)
    y = np.delete(array_y, index_, axis=0)
    index__ = np.unique(np.argwhere(y == -99)[:, 0])
    x_ = np.delete(x, index__, axis=0)
    y_ = np.delete(y, index__, axis=0)

    return x_, y_

def get_dataloader_stamp(args, normalizer='std',feature='M'):

    if args.dataset == 'M4':
        args.pred_len = M4Meta.horizons_map[args.seasonal_patterns]  # Up to M4 config
        args.seq_len = 2 * args.pred_len  # input_len = 2*pred_len
        args.label_len = args.pred_len
        args.frequency_map = M4Meta.frequency_map[args.seasonal_patterns]


        Data = Dataset_M4
        timeenc = 0 if args.embed != 'timeF' else 1

        #####  train set    ####################
        data_set = Data(
                root_path='./data/M4',
                data_path='M4',
                flag='train',
                size=[args.seq_len, args.label_len, args.pred_len],
                features='M',
                target='OT',
                timeenc=timeenc,
                freq=args.freq,
                seasonal_patterns=args.seasonal_patterns
        )

        d_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False)

        x_tra = []
        y_tra = []
        x_tra_stamp = []
        y_tra_stamp = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(d_loader):
            x_tra.append(batch_x)
            y_tra.append(batch_y)
            x_tra_stamp.append(batch_x_mark)
            y_tra_stamp.append(batch_y_mark)

        x_tra_o = torch.vstack(x_tra)
        y_tra_o = torch.vstack(y_tra)
        x_tra_stamp_o = torch.vstack(x_tra_stamp)
        y_tra_stamp_o = torch.vstack(y_tra_stamp)

        #####  test set    ####################
        data_set = Data(
            root_path='./data/M4',
            data_path='M4',
            flag='test',
            size=[args.seq_len, args.label_len, args.pred_len],
            features='M',
            target='OT',
            timeenc=timeenc,
            freq=args.freq,
            seasonal_patterns=args.seasonal_patterns
        )
        d_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False)

        x_test = []
        y_test = []
        x_test_stamp = []
        y_test_stamp = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(d_loader):
            x_test.append(batch_x)
            y_test.append(batch_y)
            x_test_stamp.append(batch_x_mark)
            y_test_stamp.append(batch_y_mark)

        x_test = torch.vstack(x_test)
        y_test = torch.vstack(y_test)
        x_test_stamp = torch.vstack(x_test_stamp)
        y_test_stamp = torch.vstack(y_test_stamp)

        #####  归一化
        data, scaler = normalize_dataset(torch.concat((x_tra_o, y_tra_o, x_test, y_test)).float(), normalizer, args.column_wise)
        n_len = int(data.shape[0]/4)
        x_tra_o = data[:n_len]
        y_tra_o = data[n_len:2*n_len]
        x_test = data[2*n_len:3*n_len]
        y_test = data[-n_len:]

        ##########  采样 ###################
        n_total = x_tra_o.shape[0]
        t_id = random.sample(range(n_total), 1000)
        heldout_index = list(set(range(n_total)).difference(set(t_id)))
        v_id = random.sample(heldout_index, 200)

        x_tra = x_tra_o[t_id]
        y_tra = y_tra_o[t_id]
        x_tra_stamp = x_tra_stamp_o[t_id]
        y_tra_stamp = y_tra_stamp_o[t_id]

        x_val = x_tra_o[v_id]
        y_val = y_tra_o[v_id]
        x_val_stamp = x_tra_stamp_o[v_id]
        y_val_stamp = y_tra_stamp_o[v_id]

        n_total = x_test.shape[0]
        test_id = random.sample(range(n_total), 500)

        x_test = x_test[test_id]
        y_test = y_test[test_id]
        x_test_stamp = x_test_stamp[test_id]
        y_test_stamp = y_test_stamp[test_id]

        train_dataloader = data_loader_stamp(x_tra, y_tra, x_tra_stamp, y_tra_stamp, args.batch_size, shuffle=True,
                                             drop_last=True)
        if len(x_val) == 0:
            val_dataloader = None
        else:
            val_dataloader = data_loader_stamp(x_val, y_val, x_val_stamp, y_val_stamp, args.batch_size, shuffle=True,
                                               drop_last=True)
        test_dataloader = data_loader_stamp(x_test, y_test, x_test_stamp, y_test_stamp, args.batch_size, shuffle=True,
                                            drop_last=True)

        return train_dataloader, val_dataloader, test_dataloader, scaler


    elif args.dataset in ['METR-LA','PEMS-BAY']:
        data, dTime = load_st_dataset(args.dataset, 'M')
        data, scaler = normalize_dataset(data.float(), 'max01', args.column_wise)

        data_train, data_val, data_test, dtime_train, dtime_val, dtime_test = split_data_by_ratio_stamp(data, dTime,
                                                                                                        args)
        x_tra, y_tra, x_tra_stamp, y_tra_stamp = Add_Window_Horizon_stamp(data_train, dtime_train, args)
        x_val, y_val, x_val_stamp, y_val_stamp = Add_Window_Horizon_stamp(data_val, dtime_train, args)
        x_test, y_test, x_test_stamp, y_test_stamp = Add_Window_Horizon_stamp(data_test, dtime_train, args)

        train_dataloader = data_loader_stamp_metr_pems(x_tra, y_tra, x_tra_stamp, y_tra_stamp, args.batch_size, shuffle=True,
                                             drop_last=True)
        if len(x_val) == 0:
            val_dataloader = None
        else:
            val_dataloader = data_loader_stamp_metr_pems(x_val, y_val, x_val_stamp, y_val_stamp, args.batch_size, shuffle=True,
                                               drop_last=True)
        test_dataloader = data_loader_stamp_metr_pems(x_test, y_test, x_test_stamp, y_test_stamp, args.batch_size, shuffle=True,
                                            drop_last=True)

        return train_dataloader, val_dataloader, test_dataloader, scaler


    else:
        data, dTime = load_st_dataset(args.dataset, feature)
        # data = data.to(args.device)
        data, scaler = normalize_dataset(data.float(), normalizer, args.column_wise)

        data_train, data_val, data_test, dtime_train, dtime_val, dtime_test = split_data_by_ratio_stamp(data, dTime, args)
        x_tra, y_tra, x_tra_stamp, y_tra_stamp = Add_Window_Horizon_stamp(data_train, dtime_train, args)
        #x_tra, y_tra = delete_outlier(x_tra_, y_tra_)
        x_val, y_val, x_val_stamp, y_val_stamp = Add_Window_Horizon_stamp(data_val, dtime_train, args)
        #x_val, y_val = delete_outlier(x_val_, y_val_)
        x_test, y_test, x_test_stamp, y_test_stamp = Add_Window_Horizon_stamp(data_test, dtime_train, args)
        #x_test, y_test = delete_outlier(x_test_, y_test_)
        print('Train:', x_tra.shape, y_tra.shape)
        print('Val:', x_val.shape, y_val.shape)
        print('Test:', x_test.shape, y_test.shape)

        train_dataloader = data_loader_stamp(x_tra, y_tra, x_tra_stamp, y_tra_stamp, args.batch_size, shuffle=True, drop_last=True)
        if len(x_val) == 0:
            val_dataloader = None
        else:
            val_dataloader = data_loader_stamp(x_val, y_val, x_val_stamp, y_val_stamp, args.batch_size, shuffle=True, drop_last=True)
        test_dataloader = data_loader_stamp(x_test, y_test, x_test_stamp, y_test_stamp, args.batch_size, shuffle=True, drop_last=True)

        return train_dataloader, val_dataloader, test_dataloader, scaler


def add_noise(xx, yy, feature, ratio):
    x_num = xx.shape[0]
    n_num = int(x_num * ratio)
    if n_num == 0:
        return xx, yy
    x_len = xx.shape[1]
    y_len = yy.shape[1]
    # #方式1, 选一部分样本，添加扰动
    # rid = random.sample(range(0,x_num),n_num)
    # if feature == 'M' or feature == 'MS':
    #     f_len = xx.shape[2]
    #     flag = torch.zeros((x_num, 1, f_len))  # 0是正常，1是噪声
    #     flag[rid] = 1
    #
    #     xx[rid,:] = xx[rid,:] + np.random.normal(0.5, 0.8 ** 2, (n_num, x_len, f_len))
    #     xx = torch.hstack((xx, flag))  # 增加1列，标识是否是噪声样本
    #     yy[rid, :] = yy[rid, :] + np.random.normal(0.5, 0.8 ** 2, (n_num, y_len, f_len))
    # elif feature == 'S':
    #     flag = torch.zeros((x_num, 1))  # 0是正常，1是噪声
    #     flag[rid, :] = 1
    #     xx[rid,:] = xx[rid,:] + np.random.normal(0.5, 0.8 ** 2, (n_num, x_len))
    #     xx = torch.hstack((xx, flag)) #增加1列，标识是否是噪声样本
    #     yy[rid, :] = yy[rid, :] + np.random.normal(0.5, 0.8 ** 2, (n_num, y_len))

    #方式2, 增添一些噪声样本
    if feature == 'M' or feature == 'MS':
        f_len = xx.shape[2]
        noiseData = np.random.normal(0.5, 0.8 ** 2, (n_num, x_len + y_len, f_len))
        flag = torch.zeros((x_num + n_num, 1, f_len))  # 0是正常，1是噪声
        flag[x_num:,0,:] = 1
    elif feature == 'S':
        noiseData = np.random.normal(0.5, 0.8 ** 2, (n_num, x_len + y_len))
        flag = torch.zeros((x_num + n_num, 1))  # 0是正常，1是噪声
        flag[x_num:, :] = 1
    noiseData = torch.tensor(noiseData)
    noiseData = noiseData.float()
    noiseData_X = noiseData[:, :x_len]
    noiseData_Y = noiseData[:, -y_len:]
    xx = torch.vstack((xx, noiseData_X))
    yy = torch.vstack((yy, noiseData_Y))
    xx = torch.hstack((xx, flag))  # 增加1列，标识是否是噪声样本

    #shuffle
    b = torch.randperm(xx.size(0))
    xx = xx[b, :]
    yy = yy[b, :]

    return xx, yy

def get_dataloader(args, normalizer='max', feature='M'):

    data, _ = load_st_dataset(args.dataset, feature)
    # data = data.to(args.device)
    data, scaler = normalize_dataset(data.float(), normalizer, args.column_wise)

    data_train, data_val, data_test = split_data_by_ratio(data, args)

    x_tra, y_tra = Add_Window_Horizon(data_train, args)

    ###############add nosie samples##########################
    # # 添加噪声数据，在数据中标识
    # x_tra, y_tra = add_noise(x_tra, y_tra, feature, args.noise_ratio)

    #x_tra, y_tra = delete_outlier(x_tra_, y_tra_)
    x_val, y_val = Add_Window_Horizon(data_val, args)
    #x_val, y_val = delete_outlier(x_val_, y_val_)
    x_test, y_test = Add_Window_Horizon(data_test, args)
    #x_test, y_test = delete_outlier(x_test_, y_test_)
    print('Train:', x_tra.shape, y_tra.shape)
    print('Val:', x_val.shape, y_val.shape)
    print('Test:', x_test.shape, y_test.shape)

    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, 32, shuffle=True, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader, scaler


if __name__ == '__main__':
    import argparse
    DATASET = 'Wind'
    parser = argparse.ArgumentParser(description='Pytorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=192, type=int)
    parser.add_argument('--horizon', default=96, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--output_dim', default=1, type=int)
    parser.add_argument('--column_wise', default=False, type=bool)
    args = parser.parse_args()
    station_name = 'JSFD001'
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, station_name, normalizer='None', single=False)
    for batch_idx, (input_data, target) in enumerate(train_dataloader):
        input_data = input_data[..., :args.input_dim]
        target = target[..., :args.output_dim]