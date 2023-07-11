#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from FL_MultiTask.models.test import test_img


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, local_bs = None):
    # def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        # self.loss_func = nn.CrossEntropyLoss(torch.Tensor(class_weight))
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=local_bs, shuffle=True)

    def train(self, net, lrr):
        net.train()
        # train and update
        # print(self.args.lr)
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = torch.optim.SGD(net.parameters(), lr=lrr, momentum=self.args.momentum)


        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdate2(object):
    def __init__(self, args, dataset=None, dataset_test=None, idxs=None):
        self.args = args
        self.dataset_test = dataset_test
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        acc_all = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            #训练完一轮后测试准确率
            net.eval()
            acc_test, loss_test = test_img(net, self.dataset_test, self.args)
            acc_all.append(acc_test)
            print("'Round {:3d}, Testing accuracy: {:.2f}".format(iter, acc_test))

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_all


class LocalUpdate_new_lossFunction(object):
    def __init__(self, args, dataset=None, idxs=None, pre_GlobalModel=None):
        self.args = args
        self.pre_globalModel = pre_GlobalModel
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        global_weight_collector = list(self.pre_globalModel.parameters())
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                delta = 0.0
                for param_index, param in enumerate(net.parameters()):
                    delta += 1 / 2 * (torch.norm((param - global_weight_collector[param_index])) ** 2)
                    # print(param)
                # print('--------------------')
                loss += delta
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

