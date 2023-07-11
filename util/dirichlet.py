'''
dirichlet.py：数据划分
________________________________________________________________________________
函数1：distribute_data_dirichlet(dataset, args, n_class=10)
作用：数据划分

输入：
    dataset：数据集名称
    args:
    n_class=10

输出：
    dict_users：（dict（list））客户端分到的数据（很多很多很多）
    pic_distribution_every_client：[[][][]...[]]每个客户端的数据分布
    sum_res：
    sum_

________________________________________________________________________________
函数2：getAllClientDataDistribution(dataset, args, n_class=10)（未用到）
作用：获取客户端针对一个任务的数据分布

输入：
    dataset：数据集名称
    args:
    n_class=10

输出：
    client_dict：客户端数据分布
________________________________________________________________________________
函数3：getTestDistribution(test_dataset)
作用：获取测试集的数据分布

输入：
    test_dataset：数据集名称

输出：
    res_all：（list）测试集数据分布

'''

#狄利克雷数据划分，将数据集划分给各个客户端

import numpy as np
import copy
import random
import utils
from options import args_parser
from utils import get_datasets
from collections import defaultdict, Counter
# from getClientQuality import getQ

#狄利克雷划分数据集
"""
    狄利克雷distribute_data_dirichlet(dataset, args, n_class=10)函数：
    功能：根据客户端数量对数据集进行划分    
    输入：数据集、各个参数信息（客户端数量）、数据集类别数
    输出：dict_users、每个客户端的数据类别分布信息、每个客户端总的数据量、所有客户端每种类型的数据量之和【重点用到中间两个】
"""
def distribute_data_dirichlet(dataset, args, n_class=10):
    np.random.seed(args.seed)
    num_clean_agents = args.num_users
    print(args.concent)
    # partition[c][i] is the fraction of samples agent i gets from class
    partition = np.random.dirichlet([args.concent] * num_clean_agents, size=n_class)
    # print(partition)

    labels_sorted = dataset.targets.sort()
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    # labels_dict[0]：所有0类的数据，此时只是将所有同类的数据放在一起了，还没有将他们划分给客户端
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
    # print(labels_dict.keys())#数据集标签

    dict_users = defaultdict(list)
    # 划分数据给客户端
    # 统计每个客户端的数据分布，即每个客户端有多少张各种类别的图片
    pic_distribution_every_client = []
    for c in range(n_class):
        # num of samples of class c in dataset 某类图片的总量
        n_classC_items = len(labels_dict[c])
        # 向所有的客户端划分图片
        pic_distribution_one_client = []
        for i in range(num_clean_agents):
            # num. of samples agent i gets from class c 第i个客户端从某类图片中分得的数量
            n_agentI_items = int(partition[c][i] * n_classC_items)
            if n_agentI_items > 0:
                pic_distribution_one_client.append(n_agentI_items)
            else:
                pic_distribution_one_client.append(0)
            dict_users[i] += labels_dict[c][:n_agentI_items]
            del labels_dict[c][:n_agentI_items]
        pic_distribution_every_client.append(pic_distribution_one_client)
        # if any class c item remains due to flooring, give em to first agent 分剩的都给第0个客户端
        dict_users[0] += labels_dict[c]
        pic_distribution_one_client[0] += len(labels_dict[c])

    pic_distribution_every_client = np.array(pic_distribution_every_client)
    pic_distribution_every_client = pic_distribution_every_client.T
    # print(dict_users)
    # print("每个客户端的数据分布：")
    # print(pic_distribution_every_client)
    sum_res = np.sum(pic_distribution_every_client, axis=1)
    sum_ = np.sum(pic_distribution_every_client, axis=0)
    # print("每个客户端的总数据量：{}".format(sum_res))

    return dict_users, pic_distribution_every_client, sum_res, sum_


# 获得所有客户端的本地所有任务的分布
def getAllClientDataDistribution(dataset, args, n_class=10):
    client_dict = []
    for data in dataset:
        dict_users, pic_distribution_every_client, sum_res, sum_ = distribute_data_dirichlet(data, args, n_class=10)
        client_dict.append(pic_distribution_every_client)
    return dict_users,client_dict

'''
    getTestDistribution(test_dataset)函数：
    功能：
    输入：测试集

'''
#获取服务器测试集数据分布
def getTestDistribution(test_dataset):
    res_all = []
    for test_data in test_dataset:
        res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for target in test_data.targets:
            res[target] += 1
        res_all.append(res)
    return res_all

'''
#20个客户端代码
def preferDistribution(dataset, args, n_class=10):
    np.random.seed(args.seed)
    # partition[c][i] is the fraction of samples agent i gets from class
    # partition = np.random.dirichlet([args.concent] * num_users, size=n_class)
    # print(partition)

    labels_sorted = dataset.targets.sort()
    print(len(dataset.targets))
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    # labels_dict[0]：所有0类的数据，此时只是将所有同类的数据放在一起了，还没有将他们划分给客户端
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
    # print(labels_dict.keys())#数据集标签

    dict_users = defaultdict(list)
    # 划分数据给客户端
    # 统计每个客户端的数据分布，即每个客户端有多少张各种类别的图片
    pic_distribution_every_client = []

    partition = np.random.dirichlet([args.concent] * 6, size=n_class)
    for c in range(n_class):
        # num of samples of class c in dataset 某类图片的总量
        n_classC_items = int(len(labels_dict[c]) * 0.8)
        # 向所有的客户端划分图片
        pic_distribution_one_client = []
        for i in range(6):
            # num. of samples agent i gets from class c 第i个客户端从某类图片中分得的数量
            n_agentI_items = int(partition[c][i] * n_classC_items)
            if n_agentI_items > 0:
                pic_distribution_one_client.append(n_agentI_items)
            else:
                pic_distribution_one_client.append(0)
            dict_users[i] += labels_dict[c][:n_agentI_items]
            del labels_dict[c][:n_agentI_items]
        pic_distribution_every_client.append(pic_distribution_one_client)


    pic_distribution_every_client = np.array(pic_distribution_every_client)
    pic_distribution_every_client_1 = pic_distribution_every_client.T
    pic_distribution_every_client = []
    partition = np.random.dirichlet([args.concent] * 14, size=n_class)
    for c in range(n_class):
        # num of samples of class c in dataset 某类图片的总量
        # n_classC_items = len(labels_dict[c]) * 0.1
        n_classC_items = len(labels_dict[c])
        # 向所有的客户端划分图片
        pic_distribution_one_client = []
        for i in range(6, 20):
            # num. of samples agent i gets from class c 第i个客户端从某类图片中分得的数量
            n_agentI_items = int(partition[c][i-6] * n_classC_items)
            if n_agentI_items > 0:
                pic_distribution_one_client.append(n_agentI_items)
            else:
                pic_distribution_one_client.append(0)
            dict_users[i] += labels_dict[c][:n_agentI_items]
            del labels_dict[c][:n_agentI_items]
        pic_distribution_every_client.append(pic_distribution_one_client)
        # if any class c item remains due to flooring, give em to first agent 分剩的都给第0个客户端
        dict_users[0] += labels_dict[c]
        pic_distribution_one_client[0] += len(labels_dict[c])

    pic_distribution_every_client = np.array(pic_distribution_every_client)
    pic_distribution_every_client_2 = pic_distribution_every_client.T

    pic_distribution_every_client = np.vstack((pic_distribution_every_client_1, pic_distribution_every_client_2))

    return dict_users, pic_distribution_every_client

def exchangeDistribution(dict_users, pic_distribution_every_client, dataset_name):
    start , end = 0, 0
    if dataset_name == 'fmnist':
        start ,end = 6, 12
    else:
        start, end = 12, 18

    for i in range(0, 6):
        temp = copy.deepcopy(dict_users[i])
        dict_users[i] = dict_users[start]
        dict_users[start] = temp

        temp = copy.deepcopy(pic_distribution_every_client[i])
        pic_distribution_every_client[i] = pic_distribution_every_client[start]
        pic_distribution_every_client[start] = temp
        start += 1

    return dict_users, pic_distribution_every_client

'''


#50个客户端代码
def preferDistribution(dataset, args, n_class=10):
    np.random.seed(args.seed)
    # partition[c][i] is the fraction of samples agent i gets from class
    # partition = np.random.dirichlet([args.concent] * num_users, size=n_class)
    # print(partition)

    labels_sorted = dataset.targets.sort()
    # print(len(dataset.targets))
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    # labels_dict[0]：所有0类的数据，此时只是将所有同类的数据放在一起了，还没有将他们划分给客户端
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
    # print(labels_dict.keys())#数据集标签

    dict_users = defaultdict(list)
    # 划分数据给客户端
    # 统计每个客户端的数据分布，即每个客户端有多少张各种类别的图片
    pic_distribution_every_client = []

    partition = np.random.dirichlet([args.concent] * 16, size=n_class)
    for c in range(n_class):
        # num of samples of class c in dataset 某类图片的总量
        n_classC_items = int(len(labels_dict[c]) * 0.6)
        # 向所有的客户端划分图片
        pic_distribution_one_client = []
        for i in range(16):
            # num. of samples agent i gets from class c 第i个客户端从某类图片中分得的数量
            n_agentI_items = int(partition[c][i] * n_classC_items)
            if n_agentI_items > 0:
                pic_distribution_one_client.append(n_agentI_items)
            else:
                pic_distribution_one_client.append(0)
            dict_users[i] += labels_dict[c][:n_agentI_items]
            del labels_dict[c][:n_agentI_items]
        pic_distribution_every_client.append(pic_distribution_one_client)


    pic_distribution_every_client = np.array(pic_distribution_every_client)
    pic_distribution_every_client_1 = pic_distribution_every_client.T
    pic_distribution_every_client = []
    partition = np.random.dirichlet([args.concent] * 34, size=n_class)
    for c in range(n_class):
        # num of samples of class c in dataset 某类图片的总量
        # n_classC_items = len(labels_dict[c]) * 0.1
        n_classC_items = len(labels_dict[c])
        # 向所有的客户端划分图片
        pic_distribution_one_client = []
        for i in range(16, 50):
            # num. of samples agent i gets from class c 第i个客户端从某类图片中分得的数量
            n_agentI_items = int(partition[c][i-16] * n_classC_items)
            if n_agentI_items > 0:
                pic_distribution_one_client.append(n_agentI_items)
            else:
                pic_distribution_one_client.append(0)
            dict_users[i] += labels_dict[c][:n_agentI_items]
            del labels_dict[c][:n_agentI_items]
        pic_distribution_every_client.append(pic_distribution_one_client)
        # if any class c item remains due to flooring, give em to first agent 分剩的都给第0个客户端
        dict_users[0] += labels_dict[c]
        pic_distribution_one_client[0] += len(labels_dict[c])

    pic_distribution_every_client = np.array(pic_distribution_every_client)
    pic_distribution_every_client_2 = pic_distribution_every_client.T

    pic_distribution_every_client = np.vstack((pic_distribution_every_client_1, pic_distribution_every_client_2))

    class_weight = []
    for i in range(len(pic_distribution_every_client)):
        temp = []
        for j in range(len(pic_distribution_every_client[i])):
            temp.append(10 * pic_distribution_every_client[i][j] / sum(pic_distribution_every_client[i]))
        class_weight.append(temp)

    return dict_users, pic_distribution_every_client,class_weight

def exchangeDistribution(dict_users, pic_distribution_every_client, dataset_name):
    start , end = 0, 0
    if dataset_name == 'fmnist':
        start ,end = 16, 32
    else:
        start, end = 32, 48

    for i in range(0, 16):
        temp = copy.deepcopy(dict_users[i])
        dict_users[i] = dict_users[start]
        dict_users[start] = temp

        temp = copy.deepcopy(pic_distribution_every_client[i])
        pic_distribution_every_client[i] = pic_distribution_every_client[start]
        pic_distribution_every_client[start] = temp
        start += 1

    class_weight = []
    for i in range(len(pic_distribution_every_client)):
        temp = []
        for j in range(len(pic_distribution_every_client[i])):
            temp.append(10*pic_distribution_every_client[i][j]/sum(pic_distribution_every_client[i]))
        class_weight.append(temp)

    return dict_users, pic_distribution_every_client, class_weight






