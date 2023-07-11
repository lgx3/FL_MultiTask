from util import dirichlet
import numpy as np
import random
from collections import defaultdict
from options import args_parser
from util.client_selection import clientSelect, getClientBudget, TaskSelected, ConfirmClientForTask, CLQM
from util.dirichlet import getAllClientDataDistribution, getTestDistribution
from util.getClientQuality import getQ
from utils import get_datasets

args = args_parser()
test_dataset_list = []
train_dataset_list = []
# dataset_name = ['fmnsit', 'cifar10']
train_dataset, test_dataset =get_datasets('fmnist')
test_dataset_list.append(test_dataset)
train_dataset_list.append(train_dataset)
train_dataset, test_dataset =get_datasets('cifar10')
test_dataset_list.append(test_dataset)
train_dataset_list.append(train_dataset)


client_dict = getAllClientDataDistribution(train_dataset_list, args, n_class=10)
testDistribution = getTestDistribution(test_dataset_list)
dic_q = getQ(client_dict, testDistribution, args)
print("客户端针对每个任务的质量为：",dic_q)#输出的为客户端所有任务的质量
budget = getClientBudget(2, 50)#获取客户端预算
print("客户端针对每个任务的报价为：",budget)

print('clientSelect测试')

# X = [1] * 20
# X_demo = [1] * 20
#
# client_selected_0, X, X_demo = clientSelect(dic_q, budget, 0, 40, X, X_demo)
#
# client_selected_1, X, X_demo = clientSelect(dic_q, budget, 1, 40, X, X_demo)
# client_selected = {0: client_selected_0, 1: client_selected_1}
# Task = [0, 1]
# task_selected = TaskSelected(X, client_selected, dic_q, Task)
# print(task_selected)
#
# S = defaultdict(list)
# for i in range(20):
#     S[i] += [0, 0]
#
# print(S)
#
# X, S = ConfirmClientForTask(S, X, task_selected, client_selected[task_selected])
# print(X)
# print(S)
#
# CLQM(dic_q, budget, args)



for i in range(5):
    print('\n****************************************** 客户端选择与支付测试——{} ******************************************'.format(i))
    CLQM(dic_q, budget, args)
    dic_q = getQ(client_dict, testDistribution, args)
    print("客户端针对每个任务的质量为：", dic_q)  # 输出的为客户端所有任务的质量
    budget = getClientBudget(2, 20)  # 获取客户端预算
    print("客户端针对每个任务的报价为：", budget)





