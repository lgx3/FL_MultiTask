'''
getClientQuality.py：获取客户端的质量Q
________________________________________________________________________________
函数1：data_distribution_distance(client_dict, test_dataset_distribution,args)
作用：根据客户端数据分布、测试集数据分布，计算emd、E_emd、Q
输入：
    client_dict：(list):一个客户端针对一个任务的数据分布[[][][][][]]
    test_dataset_distribution：(list):任务测试集数据分布
    args：
输出：
    E_emd:（list）：一个客户端的E_emd
    Q:（list）：一个客户端的质量（数据量/E_emd）

________________________________________________________________________________
函数2：getQ(client_dict, test_dataset_distribution, args):
作用：将函数1计算出来的单个任务的q汇总到一个字典里
输入：
    client_dict：(list):一个客户端针对一个任务的数据分布[[][][][][]]
    test_dataset_distribution：(list):任务测试集数据分布
    args：
输出：
    dic_Q：（dict（list））所有任务的质量
            例如：{0: [0.05432758596230657, 0.04531593186150271], 1: [0.045391643187727385, 0.06010152837008204],..., 19: [0.02662944314035432, 0.05933503960261935]}

'''

import math
from collections import defaultdict
import numpy as np

'''
________________________________________________________________________________
函数1：data_distribution_distance(client_dict, test_dataset_distribution,args)
作用：根据客户端数据分布、测试集数据分布，计算emd、E_emd、Q
输入：
    client_dict：(list):一个客户端针对一个任务的数据分布[[][][][][]]
    test_dataset_distribution：(list):任务测试集数据分布
    args：
输出：
    E_emd:（list）：一个客户端的E_emd
    Q:（list）：一个客户端的质量（数据量/E_emd）
'''
def data_distribution_distance(client_dict, test_dataset_distribution,args):
    print(client_dict)#输出客户端的数据分布
    num_clean_agents = args.num_users
    # client_data_dis = []
    # m = [0] * 10#用于存放测试集分布-客户端数据集分布
    # n = [0] * 10#用于计算emd
    Q = [0] * args.num_users#用于存放所有客户端的质量
    # q = [0] * 20
    # q = np.zeros(20)
    Emd = [0]*args.num_users
    E_emd = [0] * args.num_users

    for i in range(0,num_clean_agents):
        client_data = client_dict[i]#获取每个客户端i的数据分布，先是第一个，然后第二个，第三个，，，，
        #计算客户端数据量总和
        client_data = np.array(client_data)
        client_data_sum = np.sum(client_data)
        proportion1 = client_data / client_data_sum#计算每个客户端每种数据量占总数据量之比

        test_dataset_distribution = np.array(test_dataset_distribution)
        proportion2 = test_dataset_distribution / np.sum(test_dataset_distribution)#服务器测试集中每种数据量占总数据量之比

        emd = np.linalg.norm(proportion1 - proportion2)#客户端emd
        Emd[i] = emd
        E_emd[i] = math.exp(emd)
        q = client_data_sum / E_emd[i]
        Q[i] = q

    print('客户端emd：{}'.format(Emd))
    sum_Q = np.sum(Q)

    for i in range(args.num_users):
        Q[i] = Q[i]/sum_Q

    print('客户端Q：{}'.format(Q))

    return E_emd, Q


'''
________________________________________________________________________________
函数2：getQ(client_dict, test_dataset_distribution, args):
作用：将函数1计算出来的单个任务的q汇总到一个字典里
输入：
    client_dict：(list):一个客户端针对一个任务的数据分布[[][][][][]]
    test_dataset_distribution：(list):任务测试集数据分布
    args：
输出：
    dic_Q：（dict（list））所有任务的质量
            例如：{0: [0.05432758596230657, 0.04531593186150271], 1: [0.045391643187727385, 0.06010152837008204],..., 19: [0.02662944314035432, 0.05933503960261935]}

'''
def getQ(client_dict, test_dataset_distribution, args):
    dic_Q = defaultdict(list)
    for i in range(3):
        print("*************任务{}的客户端数据样本类型分布：*************".format(i))
        _, q_1 = data_distribution_distance(client_dict[i], test_dataset_distribution[i], args)
        print("任务{}的客户端数据样本质量Q：{}".format(i,q_1))
        client_index = 0
        for q in q_1:
            if i == 0:
                q_list = [q]
                dic_Q[client_index] += q_list
            else:
                dic_Q[client_index].append(q)
            client_index += 1
    return dic_Q


# '''不考虑'''
# # 获得所有任务的质量，不考虑数据量d:q=1/e_emd
# def getQ_without_d(client_dict, test_dataset_distribution, args):
#     dic_Q = defaultdict(list)
#     for i in range(2):
#         print("*************任务{}的客户端数据样本类型分布：*************：".format(i))
#         E_emd, _ = data_distribution_distance(client_dict[i], test_dataset_distribution[i], args)
#         client_index = 0
#         for q in E_emd:
#             if i == 0:
#                 q_list = [1/q]
#                 dic_Q[client_index] += q_list
#             else:
#                 dic_Q[client_index].append(1/q)
#             client_index += 1
#     return dic_Q
