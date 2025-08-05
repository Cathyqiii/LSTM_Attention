#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/8/25 17:50
@desc: load datasets
"""
import numpy as np
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd



def load_st_dataset(dataset, feature):

    if dataset == 'electricity':
        df_raw = pd.read_csv('./data/electricity/electricity.csv')

    if dataset == 'powerLoad':
        df_raw = pd.read_csv('./data/powerLoad/NYPowerLoad.csv')

    if dataset == 'ETTh1':
        df_raw = pd.read_csv('./data/ETTh1/ETTh1.csv')

    if dataset == 'ETTh2':
        df_raw = pd.read_csv('./data/ETTh2/ETTh2.csv')

    if dataset == 'PEMS04':
        df_raw = pd.read_csv('./data/PEMS04/PEMS04.csv')

    if dataset == 'traffic':
        df_raw = pd.read_csv('./data/traffic/traffic.csv')

    if dataset == 'exchange_rate':
        df_raw = pd.read_csv('./data/exchange_rate/exchange_rate.csv')


    target = 'OT'
    tStr = 'date'

    if feature == 'S':
        df_data = df_raw[target].values
        df_dTime = df_raw[tStr]
    elif feature == 'MS' or feature == 'M':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data].values
        df_dTime = df_raw[tStr]

    data = df_data
    dTime = df_dTime
    dTime.columns = ['date']
    print('prepare data has done!')
    return torch.tensor(data), pd.DataFrame(dTime)



if __name__ == '__main__':
    station_name = 'JSFD001'
    start_time = '20190131'
    data = load_st_dataset('powerload')
    print('prepare data has done!')