# -*- coding: utf-8 -*-
"""
@author: jimapp
@time: 2022/7/15 21:55
@desc:
"""
import os
import sys
import torch

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import numpy as np
import argparse
import configparser
from datetime import datetime
import torch.nn as nn
# from others.attn_lstm import Attn_LSTM
from others.attn_lstm_v2 import Attn_LSTM
from others.mLSTM import mLSTM
from lib.dataloader import get_dataloader_stamp
import time
from lib.BasicTrainer_sw import Trainer
from torch.utils.data.dataloader import DataLoader
from lib.dataloader import data_loader
from TSlib.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from TSlib.exp.exp_imputation import Exp_Imputation
# from TSlib.exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from TSlib.exp.exp_anomaly_detection import Exp_Anomaly_Detection
from TSlib.exp.exp_classification import Exp_Classification
import random
from lib.addnoise import add_noise2
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 自定义Huber损失函数
class HuberLoss(nn.Module):
    def __init__(self, delta=0.5):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, preds, labels):
        diff = torch.abs(preds - labels)
        loss = torch.where(diff < self.delta,
                           0.5 * diff ** 2,
                           self.delta * (diff - 0.5 * self.delta))
        return torch.mean(loss)


today = time.time()
seed_num = 2024
for tt in range(3):
    setup_seed(seed_num + tt * 100)
    print(f'--------------{tt}--------------------inter----')
    for DATASET in ['powerLoad']:
        print(DATASET)
        Mode = 'Train'  # Train or test
        DEBUG = 'True'
        optim = 'adam'  # 使用Adam优化器
        DEVICE = 'cpu'
        MODEL = 'lstm-att'
        ktype = 'normal'
        noise_ratio = 0  # 0 0.3
        feature = 'S'
        task_name = 'long_term_forecast'
        finish_time = 1662287958.9541638
        # config_file
        config_file = 'configs/{}/{}.conf'.format(DATASET, MODEL)
        config = configparser.ConfigParser()
        config.read(config_file)

        from lib.metrics import MAE_torch


        def masked_mae_loss(scaler, mask_value):
            def loss(preds, labels):
                if scaler:
                    preds = scaler.inverse_transform(preds)
                    labels = scaler.inverse_transform(labels)
                mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
                return mae

            return loss


        # parser
        args = argparse.ArgumentParser(description='arguments')

        # basic config
        args.add_argument('--task_name', type=str, default=task_name,
                          help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        args.add_argument('--is_training', type=int, default=1, help='status')
        args.add_argument('--model_id', type=str, default='test', help='model id')
        args.add_argument('--model', type=str, default=MODEL,
                          help='model name, options: [Autoformer, Transformer, TimesNet]')
        args.add_argument('--dataset', default=DATASET, type=str)
        args.add_argument('--mode', default=Mode, type=str)
        args.add_argument('--optim', default=optim, type=str)
        args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
        # args.add_argument('--debug', default=DEBUG, type=eval)
        # args.add_argument('--model', default=MODEL, type=str)
        # args.add_argument('--cuda', default=True, type=bool)

        # data
        args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
        args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
        args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
        args.add_argument('--stamp', default=config['data']['stamp'], type=bool)
        args.add_argument('--freq', type=str, default=config['data']['freq'],
                          help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        # model
        args.add_argument('--top_k', type=int, default=config['model']['top_k'], help='for TimesBlock')
        args.add_argument('--num_kernels', type=int, default=config['model']['num_kernels'], help='for Inception')
        args.add_argument('--dec_in', default=config['model']['dec_in'], type=int)
        args.add_argument('--enc_in', default=config['model']['enc_in'], type=int)
        args.add_argument('--c_out', type=int, default=config['model']['c_out'], help='output size')
        args.add_argument('--d_model', default=config['model']['d_model'], type=int)
        args.add_argument('--n_heads', type=int, default=config['model']['n_heads'], help='num of heads')
        args.add_argument('--d_ff', type=int, default=config['model']['d_ff'], help='dimension of fcn')
        args.add_argument('--moving_avg', type=int, default=config['model']['moving_avg'],
                          help='window size of moving average')
        args.add_argument('--factor', type=int, default=config['model']['factor'], help='attn factor')
        args.add_argument('--embed', type=str, default=config['model']['embed'],
                          help='time features encoding, options:[timeF, fixed, learned]')
        args.add_argument('--dropout', type=float, default=config['model']['dropout'], help='dropout')
        args.add_argument('--timeenc', type=int, default=config['model']['timeenc'], help='dropout')

        # args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
        # args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
        args.add_argument('--e_layers', type=int, default=config['model']['e_layers'], help='num of encoder layers')
        args.add_argument('--d_layers', type=int, default=config['model']['d_layers'], help='num of decoder layers')

        # 增强模型容量
        args.add_argument('--num_layers', default=3, type=int)  # 增加LSTM层数
        args.add_argument('--hidden_size', default=128, type=int)  # 增加隐藏单元

        args.add_argument('--column_wise', default=config['model']['column_wise'], type=bool)
        args.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        args.add_argument('--activation', type=str, default='gelu', help='activation')
        args.add_argument('--distil', action='store_false',
                          help='whether to use distilling in encoder, using this argument means not using distilling',
                          default=True)
        # train
        args.add_argument('--loss_func', default='huber', type=str)  # 使用Huber损失
        args.add_argument('--seed', default=config['train']['seed'], type=int)
        args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
        args.add_argument('--epochs', default=100, type=int)  # 增加训练轮次
        args.add_argument('--lr_init', default=0.001, type=float)  # 降低初始学习率
        args.add_argument('--lr_decay', default=True, type=eval)
        args.add_argument('--lr_decay_rate', default=0.95, type=float)
        args.add_argument('--lr_decay_step', default='20,40,60,80', type=str)  # 更细粒度的学习率衰减
        args.add_argument('--early_stop', default=True, type=eval)
        args.add_argument('--early_stop_patience', default=15, type=int)  # 增加早停耐心值
        args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
        args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
        args.add_argument('--teacher_forcing', default=config['train']['teacher_forcing'], type=eval)
        args.add_argument('--tf_decay_steps', default=config['train']['tf_decay_steps'], type=int,
                          help='teacher forcing decay steps')
        args.add_argument('--real_value', default=config['train']['real_value'], type=eval,
                          help='use real value for loss calculation')
        # test
        args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
        args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
        # log
        # args.add_argument('--log_dir', default='./', type=str)
        args.add_argument('--log_step', default=config['log']['log_step'], type=int)
        # args.add_argument('--plot', default=config['log']['plot'], type=eval)

        # forecasting task
        args.add_argument('--lag', default=config['data']['lag'], type=int)
        args.add_argument('--step', default=config['data']['step'], type=int)
        args.add_argument('--window', default=config['data']['window'], type=int)
        args.add_argument('--interval', default=config['data']['interval'], type=int)
        args.add_argument('--horizon', default=config['data']['horizon'], type=int)
        args.add_argument('--label_len', type=int, default=config['data']['label_len'], help='start token length')

        # GPU
        args.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        args.add_argument('--gpu', type=int, default=0, help='gpu')
        args.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        args.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        args = args.parse_args()

        ######################GPU#################################
        if args.device == 'cpu':
            args.device = 'cpu'
            args.use_gpu = False
        elif args.device == 'cuda:0':
            if torch.cuda.is_available():
                torch.cuda.set_device(int(args.device[5]))
                args.use_gpu = True
            else:
                args.device = 'cpu'
                args.use_gpu = False
        ######################read data#################################

        args.noise_ratio = noise_ratio

        train_loader_orig, val_loader, test_loader, scaler = get_dataloader_stamp(args,
                                                                                  normalizer=args.normalizer,
                                                                                  feature=feature)

        args.batch_size = train_loader_orig.batch_size
        # init model
        args.stamp = True

        ###############add nosie samples##########################

        if noise_ratio == 0:
            train_loader = train_loader_orig
        elif noise_ratio != 0:

            train_set = train_loader_orig.dataset
            train_set = DataLoader(dataset=train_set, batch_size=1, shuffle=False)
            trainX = []
            trainY = []
            for index, (X, y_true) in enumerate(train_set):
                trainX.append(X.cpu().numpy())
                trainY.append(y_true.cpu().numpy())

            newX, newY, bidx = add_noise2(torch.tensor(trainX).squeeze(), torch.tensor(trainY).squeeze(),
                                          args.noise_ratio, args.device)

            train_loader = data_loader(newX, newY, args.batch_size, shuffle=False, drop_last=True)

        args.ktype = ktype

        for hh in [4]:
            if hh == 0:
                args.horizon = 1
                args.window = 1
            else:
                args.horizon = hh
                args.window = 6

            ####################################################
            args.seq_len = args.interval * args.lag
            args.pred_len = args.window * args.horizon
            args.noise_ratio = noise_ratio

            #######################################################
            current_time = datetime.now().strftime('%Y%m%d%H%M%S')
            current_dir = os.path.dirname(os.path.realpath(__file__))
            log_dir = os.path.join(current_dir, 'experiments', args.dataset, current_time)
            args.log_dir = log_dir

            if feature == 'S':
                # args.en_input_dim = 1
                args.enc_in = 1
                args.dec_in = 1
                args.c_out = 1
            #######################################################
            if args.task_name == 'long_term_forecast':
                Exp = Exp_Long_Term_Forecast
            # elif args.task_name == 'short_term_forecast':
            #     Exp = Exp_Short_Term_Forecast
            elif args.task_name == 'imputation':
                Exp = Exp_Imputation
            elif args.task_name == 'anomaly_detection':
                Exp = Exp_Anomaly_Detection
            elif args.task_name == 'classification':
                Exp = Exp_Classification
            else:
                Exp = Exp_Long_Term_Forecast

            if MODEL in ['TimesNet', 'Autoformer', 'Transformer', 'Nonstationary_Transformer', 'DLinear',
                         'FEDformer', 'Informer', 'LightTS', 'Reformer', 'ETSformer', 'PatchTST', 'Pyraformer',
                         'MICN', 'Crossformer', 'FiLM']:
                exp = Exp(args)  # set experiments
                # exp.model_dict[args.model].Model(args).float()
                model = exp.model
            elif MODEL in ['lstm-att']:
                model = Attn_LSTM(args)
            elif MODEL in ['mLSTM']:
                model = mLSTM(args)

            # 改进的权重初始化
            for name, param in model.named_parameters():
                if 'weight' in name:
                    if len(param.shape) > 1:
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                    else:
                        nn.init.normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.1)

            # 改进的损失函数
            if args.loss_func == 'mask_mae':
                loss = masked_mae_loss(scaler, mask_value=0.0)
            elif args.loss_func == 'huber':
                loss = HuberLoss(delta=0.5).to(args.device)  # Huber损失
            elif args.loss_func == 'mae':
                loss = torch.nn.L1Loss().to(args.device)
            elif args.loss_func == 'mse':
                loss = torch.nn.MSELoss().to(args.device)
            else:
                loss = HuberLoss(delta=0.5).to(args.device)  # 默认使用Huber

            quality = torch.tensor(1)
            if ktype in ['normal']:
                # 使用Adam优化器
                if args.optim == 'adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init,
                                                 weight_decay=1e-4)  # 添加L2正则化
                elif args.optim == 'sgd':
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=0.9)
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)

                # 使用余弦退火学习率调度器
                lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

                # 改进的训练器初始化
                trainer = Trainer(ktype, model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                                  args, lr_scheduler=lr_scheduler)

                if args.mode == 'Train':
                    trainer.train()
                elif args.mode == 'test':
                    model.load_state_dict(torch.load('./experiments/best_model_{}.pth'.format(finish_time)))
                    print("Load saved model")
                    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger, finish_time=finish_time)

                output_path = './ns_results/%02d' % (seed_num)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                filename = f'{args.dataset}_{args.model}_{ktype}_{optim}_{args.early_stop_patience}_ns_{args.noise_ratio}_{args.horizon * args.window}_{today}.csv'

                if tt == 0:
                    trainer.results.to_csv(
                        f'{output_path}/{filename}',
                        mode='a',
                        header=True
                    )
                else:
                    trainer.results.to_csv(
                        f'{output_path}/{filename}',
                        mode='a',
                        header=False
                    )
            else:
                pass

        del train_loader, val_loader, test_loader, scaler
        torch.cuda.empty_cache()