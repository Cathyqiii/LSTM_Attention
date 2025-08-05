#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2022/7/1 22:05
@desc:
"""
import torch
import numpy as np

def perform_bernoulli_trials(p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n = p.shape[0]
    r_list = []
    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()
        try:
            # If less than p, it's a success so add one to n_success
            if random_number > p[i]:
                r_list.append(1)
            else:
                r_list.append(p[i].detach().cpu().item())
        except Exception:
            r_list.append(1)

    return torch.tensor(r_list).to(p.device)

def store_grad(pp, grads, grad_dims, simple_ev, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1

    #weight = simple_ev[0, tid].to(grads.device)
    weight = simple_ev[tid].to(grads.device)
    grads[:, tid] = grads[:, tid] * weight
    #grads[:, tid] = grads[:, tid]/torch.linalg.norm(grads[:, tid].float()).reshape(-1,1)


def store_grad_nosadc(pp, grads, grad_dims, simple_ev, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1



def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def entropyValue2(decoder_attentions):
    #Calculate entropy by row
    da = decoder_attentions.detach()
    # da = da.squeeze()
    col_sum = da.sum(1)
    row_sum = col_sum.sum(1)
    row_sum = row_sum.unsqueeze(1)
    prob = col_sum / row_sum
    log_prob = torch.log2(prob)
    multi = log_prob * prob
    e_value = -multi.sum(1)
    return e_value



