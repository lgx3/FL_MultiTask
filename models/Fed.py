#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

# 这个联邦平均算法是不是有点问题呢? 我改过了
def FedAvg(w, p,idxs_users):
    # print('idxs_users={}'.format(idxs_users))
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * p[idxs_users[0]]
        # print('当key是{}时'.format(k))
        for i in range(1, len(w)):
            # print('计算{}号客户端的参数'.format(idxs_users[i]))
            w_avg[k] += w[i][k] * p[idxs_users[i]]
        # w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvgMatch(w, p, idxs_users):
    # print('idxs_users={}'.format(idxs_users))
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * (p[idxs_users[0][0]] + p[idxs_users[0][1]])
        # print('当key是{}时'.format(k))
        for i in range(1, len(w)):
            # print('计算{}号客户端的参数，p值是{}和{}'.format(idxs_users[i], p[idxs_users[i][0]], p[idxs_users[i][1]]))
            w_avg[k] += w[i][k] * (p[idxs_users[i][0]] + p[idxs_users[i][1]])
        # w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
