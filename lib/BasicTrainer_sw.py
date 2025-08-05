#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2022/7/1 22:05
@desc:
"""
import copy
import torch
import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from model.admm import proxADMM
import gpytorch
from lib.logger import get_logger
from lib.metrics import All_Metrics
from lib.simple_Weight import entropyValue2, perform_bernoulli_trials, store_grad, overwrite_grad
import pandas as pd
from scipy.optimize import minimize
# from lib.GaussianProcess import GPR, ExactGPModel

class Trainer(object):
    def __init__(self, ktype, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.ktype = ktype
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.device = args.device
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.grad_dims = []
        self.confict_num = 0
        self.num_iterations = 10
        self.model = model.to(args.device)

        self.batach_atten_loss = np.zeros((1,2, self.args.batch_size,))
        if self.args.dataset == 'traffic':
            self.batach_attweight = np.zeros((1,self.args.batch_size, 576))
        else:
            self.batach_attweight = np.zeros((1, self.args.batch_size, 288))
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        # if os.path.isdir(args.log_dir) == False and not args.debug:
        #     os.makedirs(args.log_dir, exist_ok=True)
        # self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        # self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        # print('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

        self.results = pd.DataFrame(columns=['dataset', 'MAE', 'RMSE', 'MAPE', 'steps', 'bestModel'])

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (x_train, y_true) in enumerate(val_dataloader):

                x_train = x_train.to(self.device).float()
                y_true = y_true.to(self.device).float()

                y_true = y_true[:,
                             self.args.label_len:self.args.label_len + self.args.window * self.args.horizon]  # 有label长度
                label = y_true[:, :, 0]  # 第2维是时间戳

                x_enc = x_train[:, :, 0].unsqueeze(2)
                x_mark_enc = x_train[:, :, 1:]
                x_dec = y_true[:, :, 0].unsqueeze(2)
                x_mark_dec = y_true[:, :, 1:]

                # seq_last = x_enc[:, -1, :].unsqueeze(1)
                # x_enc = x_enc - seq_last
                if self.ktype in ['sadc', 'cagrad', 'pcgrad']:
                    output, decoder_attentions = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                else:
                    output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

                # output = output + seq_last
                output = output.squeeze(2)


                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)

                loss = self.loss(output, label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        if epoch % self.args.log_step == 0:
            # self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
            print('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return torch.allclose(a, a.T, rtol=rtol, atol=atol)

    def cagrad_multi(self, grads, c=0.5):
        grads = grads.numpy()
        g0 = np.mean(grads, axis=1)
        g0_norm = np.linalg.norm(g0)
        phi = c * g0_norm

        K = grads.shape[1]

        # obj1 = lambda x: np.mean(x*grads, axis=0)
        # con1 = lambda x: x.sum()-1
        def obj(x):
            gw = np.mean(x * grads, axis=1)
            gw_norm = np.linalg.norm(gw)
            t1 = gw.T @ g0
            t2 = phi * gw_norm
            return t1 + t2

        def con(x):
            J = x.sum() - 1
            return J

        cons = ({'type': 'eq', 'fun': con},
                )

        bnds = tuple((0, 1) for i in range(K))

        x0 = np.zeros(K)
        res = minimize(obj, x0, constraints=cons, method='SLSQP', bounds=bnds)

        x = res.x

        gw = x * grads
        gw = np.mean(gw, axis=1)
        gw_norm = np.linalg.norm(gw)

        lmbda = phi / (gw_norm + 1e-4)
        g = g0 + lmbda * gw
        return torch.tensor(g * c)

    def grp_train(self, model, likelihood, training_iter, train_x, train_y):
        #模式设置
        model.train()
        likelihood.train()

        # 优化器Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y).sum()
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            optimizer.step()

        return model, likelihood

    # def evaluate(self, data, label):
    #     # 估计评估
    #     dim = data.shape
    #     train_x = data[:, :-1, :-1].contiguous()
    #     train_y = data[:, :-1, -1].contiguous()
    #     test_x = label[:, :, :-1].contiguous()
    #     test_y = label[:, :, -1].contiguous()
    #     f_dim = dim[-1] - 1
    #     batch = dim[0]
    #
    #     likelihood = gpytorch.likelihoods.GaussianLikelihood(
    #         noise_constraint=gpytorch.constraints.GreaterThan(1e-11))
    #     likelihood.register_prior("noise_prior", gpytorch.priors.HorseshoePrior(0.1), "noise")
    #     likelihood.noise = 1e-2
    #     model = ExactGPModel(train_x, train_y, likelihood, batch)
    #     model, likelihood = self.grp_train(model, likelihood, 5, train_x, train_y)
    #
    #     # Get into evaluation (predictive posterior) mode
    #     model.eval()
    #     likelihood.eval()
    #     # Test points are regularly spaced along [0,1]
    #     # Make predictions by feeding model through likelihood
    #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #         # observed_pred = likelihood(model(test_x)) #预测
    #         #获得函数
    #         pred = model(test_x)
    #     mu = pred.mean
    #     varz = pred.variance
    #
    #     # from matplotlib import pyplot as plt
    #     # import matplotlib
    #     # matplotlib.use('TkAgg')
    #     # _task = 0
    #     # plt.figure()
    #     # plt.plot(test_y[_task].detach().cpu().numpy(), 'r*-')
    #     # plt.plot(mu[_task].detach().cpu().numpy(),'b')
    #     # plt.legend(['Test Data', 'Mean'], fontsize=16)
    #     # plt.show()
    #
    #     return  mu, varz

    def train_epoch(self, epoch):
        # global optimizer
        self.model.train()
        total_loss = 0
        train_epoch_loss = 0

        alo_time = []
        scsq_time = []
        conf_time = []

        if self.args.optim == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.args.lr_init)
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.args.lr_init, eps=1.0e-8,
                                         weight_decay=0, amsgrad=False)

        for batch_idx, (x_train, y_true) in enumerate(self.train_loader):

            x_train = x_train.to(self.device).float()
            y_true = y_true.to(self.device).float()

            y_true = y_true[:,
                         self.args.label_len:self.args.label_len + self.args.window * self.args.horizon]  # 有label长度
            label = y_true[:, :, 0]  # 第2维是时间戳

            x_enc = x_train[:, :, 0].unsqueeze(2)
            x_mark_enc = x_train[:, :, 1:]
            x_dec = y_true[:, :, 0].unsqueeze(2)
            x_mark_dec = y_true[:, :, 1:]

            # seq_last = x_enc[:,-1,:].unsqueeze(1)
            # x_enc = x_enc - seq_last

            if self.ktype in ['sadc', 'cagrad', 'pcgrad']:
                output, decoder_attentions = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # output = output + seq_last
            output = output.squeeze(2)


            self.grad_dims.clear()
            for param in self.model.parameters():
                self.grad_dims.append(param.data.numel())

            dim = x_train.shape
            self.grads = torch.Tensor(sum(self.grad_dims), dim[0])

            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 0

            if self.args.real_value:
                label = self.scaler.inverse_transform(label)

            if self.ktype != 'normal':
                simple_ev = entropyValue2(decoder_attentions) #[2,64]
                smax = simple_ev.max()
                smin = simple_ev.min()
                sev_scale = (simple_ev - smin)/(smax - smin)

                ### noise identification
                # sorted_vals = torch.argsort(sev_scale)
                # prop_info = []
                # for prop in [5,10,20,40]:
                #     oridx = []
                #     for ii in range(prop):
                #         oridx.append(bidx[sorted_vals[ii].item()])
                #     oridxnp = np.array(oridx)
                #     prop_info.append(sum(oridxnp > 64))
                # print(f'5%,{prop_info[0]},10%,{prop_info[1]},15%,{prop_info[2]},20%,{prop_info[3]}')
                ###########################
                r_list = perform_bernoulli_trials(sev_scale)

                batch_loss = []
                for i in range(dim[0]):
                    optimizer.zero_grad()
                    b_item_loss = self.loss(output[i,:], label[i,:])
                    b_item_loss.backward(retain_graph=True)
                    batch_loss.append(b_item_loss.detach().cpu().item())

                    store_grad(self.model.parameters, self.grads, self.grad_dims, r_list, i)

                if self.ktype == 'pcgrad':
                    del_times = 10
                    alo_time_start = time.time()
                    while (del_times > 0):
                        dotp = torch.mm(self.grads.T, self.grads)
                        dotind = torch.nonzero(dotp < 0).squeeze()
                        #dotind = torch.argwhere(dotp < 0)
                        if len(dotind) > 0:
                            self.confict_num = self.confict_num + 1
                            if dotind.shape[0] == 2:
                                if self.check_symmetric(dotind):
                                    dot_i = dotind[0, :].unsqueeze(0)
                            elif dotind.shape[0] == 1:
                                dot_i = dotind.unsqueeze(0)
                            else:
                                # Find the top right index
                                dot_list = [i if dotind[i, 0] < dotind[i, 1] else 0 for i in range(dotind.shape[0])]
                                # Get the location of the conflict vector,dot_i[0] dot_i[1]
                                dot_d = [i for i, x in enumerate(dot_list) if x > 0]
                                dot_i = dotind[dot_d]

                            ai = self.grads.T[dot_i[:, 0]]
                            ai_o = copy.deepcopy(ai)
                            aj = self.grads.T[dot_i[:, 1]]
                            aj_o = copy.deepcopy(aj)
                            # update: ai,aj
                            ijd = ai_o.mm(aj_o.T).diag()
                            vlen_j = (aj_o * aj_o).sum(1)
                            vlen_i = (ai_o * ai_o).sum(1)

                            ijd = ijd.reshape(-1, 1)
                            pvlen_i = vlen_i.reshape(-1, 1)
                            pvlen_j = vlen_j.reshape(-1, 1)

                            ai = ai - ijd * aj_o / pvlen_j
                            aj = aj - ijd * ai_o / pvlen_i
                            # update
                            self.grads.T[dot_i[:, 0]] = ai
                            self.grads.T[dot_i[:, 1]] = aj

                            del_times = del_times - 1
                        else:
                            break
                    alo_time_end = time.time()
                    alo_time.append(alo_time_end - alo_time_start)
                    conf_time.append(10 - del_times)

                    # average
                    self.grads = self.grads.float().mean(1)
                elif self.ktype == 'cagrad':

                    alo_time_start = time.time()
                    self.grads = self.cagrad_multi(self.grads)
                    alo_time_end = time.time()
                    alo_time.append(alo_time_end - alo_time_start)

                elif self.ktype == 'sadc':

                    tempG = self.grads.transpose(0, 1).detach().cpu().numpy()
                    alo_time_start = time.time()
                    self.admm = proxADMM(tempG)
                    last_x = np.zeros_like(np.mean(tempG, axis=0))
                    for i in range(0, self.num_iterations):
                        self.admm.step_iterative()
                        #conf = np.dot(tempG, self.admm.getParams())
                        # print('{}Val:'.format(i), conf)
                        # print(-conf.sum())
                        current_x = self.admm.getParams()
                        diff = np.abs(np.sum(last_x - current_x))
                        last_x = current_x
                        if diff < 1e-5:
                            scsq_time.append(i)
                            #print(f'early stop at {i}-th iterations')
                            break
                    alo_time_end = time.time()
                    alo_time.append(alo_time_end - alo_time_start)
                    #print(alo_time_end - alo_time_start)
                    #print(f'current x = {admm.getParams()}')
                    self.grads = torch.tensor(self.admm.getParams().reshape(-1,1))

                # copy gradients back
                overwrite_grad(self.model.parameters, self.grads, self.grad_dims)

                self.optimizer.step()
                total_loss += torch.tensor(batch_loss).sum()

            else:
                # normal approach
                b_loss = self.loss(output, label)
                b_loss.backward()

                self.optimizer.step()
                total_loss += b_loss.item()

        #log information
        if epoch % self.args.log_step == 0:
            print(f"conflict num {self.confict_num}")
            #self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
            #    epoch, batch_idx, self.train_per_epoch, loss.item()))
            train_epoch_loss = total_loss/self.train_per_epoch
            # self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))
            print(
                '**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss,
                                                                                       teacher_forcing_ratio))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss, alo_time, scsq_time


    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        alotimes = []
        scsqtimes = []
        for epoch in range(1, self.args.epochs + 1):
            # print(epoch)
            #epoch_time = time.time()
            train_epoch_loss, alo_time, scsq_time = self.train_epoch(epoch)
            if epoch < 50:
                alotimes.extend(alo_time)
                scsqtimes.extend(scsq_time)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if best_loss - val_epoch_loss > 0.0001:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    # self.logger.info("Validation performance didn\'t improve for {} epochs. "
                    #                 "Training stops.".format(self.args.early_stop_patience))
                    print("Validation performance didn\'t improve for {} epochs. "
                          "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                '''
                if epoch % self.args.log_step == 0:
                    self.logger.info('*********************************Current best model saved!')
                '''
                best_model = copy.deepcopy(self.model.state_dict())

        finish_time = time.time()
        training_time = finish_time - start_time
        # self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        print("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # if self.args.dataset == 'powerLoad':
        #     np.save(f'./imgPower/attloss.npy', self.batach_atten_loss)
        #     np.save(f'./imgPower/attweight.npy', self.batach_attweight)
        # elif self.args.dataset == 'traffic':
        #     np.save(f'./imgTraffic/attloss.npy', self.batach_atten_loss)
        #     np.save(f'./imgTraffic/attweight.npy', self.batach_attweight)
        # elif self.args.dataset == 'ETDataset':
        #     np.save(f'./imgEDT/attloss.npy', self.batach_atten_loss)
        #     np.save(f'./imgEDT/attweight.npy', self.batach_attweight)
        #save the best model to file
        #if not self.args.debug:

        # torch.save(best_model, './experiments/best_model_{}_{}_{}_{}.pth'.format(self.args.model, self.args.dataset, self.args.horizon,finish_time))

        #self.logger.info("Saving current best model to " + self.best_path)

        self.alotimes = pd.DataFrame(alotimes)
        self.scsqtimes = pd.DataFrame(scsqtimes)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        mae, rmse, mape = self.test(self.model, self.args, self.test_loader, self.scaler, finish_time=finish_time)

        pf = pd.DataFrame({
            'dataset': [self.args.dataset],
            'MAE': [mae],
            'RMSE': [rmse],
            'MAPE': [mape],
            'steps': [epoch],
            'bestModel': [
                './experiments/best_model_{}_{}_{}_{}.pth'.format(self.args.model, self.args.dataset, self.args.horizon,
                                                                  finish_time)],
        })
        # self.results = self.results.append(pf, ignore_index=True)
        self.results = pd.concat([self.results, pf])

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        # self.logger.info("Saving current best model to " + self.best_path)
        print("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, path=None, finish_time=0):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred_t = []
        y_true_t = []
        total_attentions = []
        total_datas = []

        with torch.no_grad():
            for batch_idx, (x_train, y_true) in enumerate(data_loader):
                # if args.stamp:
                #     target = target[:, -args.window * args.horizon:]
                #     label = target[:, :, 0]
                # else:
                #     target = target[:, :args.window * args.horizon]
                #     label = target
                x_train = x_train.to(args.device).float()
                y_true = y_true.to(args.device).float()

                y_true = y_true[:,
                             args.label_len:args.label_len + args.window * args.horizon]  # 有label长度
                label = y_true[:, :, 0]  # 第2维是时间戳

                x_enc = x_train[:, :, 0].unsqueeze(2)
                x_mark_enc = x_train[:, :, 1:]
                x_dec = y_true[:, :, 0].unsqueeze(2)
                x_mark_dec = y_true[:, :, 1:]

                # seq_last = x_enc[:, -1, :].unsqueeze(1)
                # x_enc = x_enc - seq_last

                if args.ktype in ['sadc', 'cagrad', 'pcgrad','grand', 'vog','sgd-influence', 'd-shapely']:
                    output, decoder_attentions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                else:
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

                # output = output + seq_last
                output = output.squeeze(2)


                y_true_t.append(label)
                y_pred_t.append(output)

        y_pred_t = torch.cat(y_pred_t, dim=0)
        y_true_t = torch.cat(y_true_t, dim=0)
        if args.real_value:
            y_true_t = scaler.inverse_transform(y_true_t)
            y_pred_t = scaler.inverse_transform(y_pred_t)

        print_time = finish_time
        # if not os.path.exists(
        #         './results/{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.horizon,
        #                                              args.window,
        #                                              print_time)):
        #     try:
        #         os.makedirs(
        #             './results/{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.horizon,
        #                                                  args.window, print_time))
        #         np.save(
        #             './results/{}_{}_{}_{}_{}/{}_true.npy'.format(args.dataset, args.model, args.horizon,
        #                                                              args.window, print_time,
        #                                                              args.dataset), y_true.cpu().numpy())
        #         np.save(
        #             './results/{}_{}_{}_{}_{}/{}_pred.npy'.format(args.dataset, args.model, args.horizon,
        #                                                              args.window, print_time,
        #                                                              args.dataset), y_pred.cpu().numpy())
        #         if args.model in ['shapeSim']:
        #             np.save('./results/{}_{}_{}_{}_{}/{}_data.npy'.format(args.dataset, args.model, args.horizon,
        #                                                                      args.window,
        #                                                                      print_time, args.dataset),
        #                     torch.stack(total_datas, dim=0).cpu().numpy())
        #             np.save(
        #                 './results/{}_{}_{}_{}_{}/{}_attention.npy'.format(args.dataset, args.model, args.horizon,
        #                                                                        args.window,
        #                                                                       print_time, args.dataset),
        #                 torch.stack(total_attentions, dim=0).cpu().numpy())
        #     except:
        #         print('attention error')

        r_metrics = []
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred_t[:, t], y_true_t[:, t],
                                          args.mae_thresh, args.mape_thresh)
            print("Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape * 100))
            r_metrics.append([t, mae.cpu().item(), rmse.cpu().item(), mape.cpu().item()])


        mae, rmse, mape = All_Metrics(y_pred_t, y_true_t, args.mae_thresh, args.mape_thresh)
        # logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
        #     mae, rmse, mape * 100))
        print("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
            mae, rmse, mape * 100))
        # if not os.path.exists(
        #         './results/{}_{}_{}_{}_{}/{}_metrics.npy'.format(args.dataset, args.model, args.horizon,
        #                                                             args.window, print_time,
        #                                                             args.dataset)):
        #     np.save('./results/{}_{}_{}_{}_{}/{}_metrics.npy'.format(args.dataset, args.model, args.horizon,
        #                                                                 args.window, print_time,
        #                                                                 args.dataset),
        #             torch.tensor([mae, rmse, mape]).cpu().numpy())
        # # r_metrics = np.array(r_metrics)
        return mae.cpu().item(), rmse.cpu().item(), mape.cpu().item()

    @staticmethod
    def test_gbrt(model, args, train_loader, test_loader, scaler, logger, path=None, finish_time=0):
        y_pred = []
        y_true = []
        total_data = []
        total_target = []

        for batch_idx, (data, target) in enumerate(train_loader):
            total_data.append(data)
            total_target.append(target)

        total_data = torch.cat(total_data, dim=0)
        total_target = torch.cat(total_target, dim=0)

        label = total_target[:, :args.window * args.horizon]
        clf = model.train(total_data.cpu().numpy(), label.cpu().numpy())

        for i, (data, target) in enumerate(test_loader):
            pred_one = torch.zeros(data.shape[0], args.window * args.horizon)

            output = model(data.cpu().numpy(), clf, pred_one.cpu().numpy())

            label = target[:, :args.window * args.horizon]

            y_true.append(label)
            y_pred.append(torch.tensor(output).to(label.device))

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        if args.real_value:
            y_true = scaler.inverse_transform(y_true)
            y_pred = scaler.inverse_transform(y_pred)

        print_time = finish_time
        if not os.path.exists('./results/{}_GBRT_{}_{}'.format(args.dataset, args.horizon,print_time)):
            os.makedirs('./results/{}_GBRT_{}_{}'.format(args.dataset, args.horizon, print_time))
            np.save('./results/{}_GBRT_{}_{}/{}_true.npy'.format(args.dataset, args.horizon, print_time, args.dataset),
                    y_true.cpu().numpy())
            np.save('./results/{}_GBRT_{}_{}/{}_pred.npy'.format(args.dataset, args.horizon, print_time, args.dataset),
                    y_pred.cpu().numpy())

        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred[:, t], y_true[:, t],
                                          args.mae_thresh, args.mape_thresh)
            # logger.info("GBRT=========Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
            #     t + 1, mae, rmse, mape * 100))
            print("GBRT=========Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape * 100))
        mae, rmse, mape = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        # logger.info("GBRT=========verage Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
        #     mae, rmse, mape * 100))
        print("GBRT=========verage Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
            mae, rmse, mape * 100))
        if not os.path.exists(
                './results/{}_GBRT_{}_{}/{}_metrics.npy'.format(args.dataset, args.horizon, print_time, args.dataset)):
            np.save(
                './results/{}_GBRT_{}_{}/{}_metrics.npy'.format(args.dataset, args.horizon, print_time, args.dataset),
                torch.tensor([mae, rmse, mape]).cpu().numpy())

        return mae.cpu().item(), rmse.cpu().item(), mape.cpu().item()

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))