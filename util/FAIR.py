
import numpy as np
from collections import defaultdict

#CLQM选择：输入：  质量  、报价、预算

"""
计算获取所有任务中所有客户端的质量计算*
FAIR方法的获取质量：
（1）获取客户端训练的损失、测试损失，得到m
（2）质量 = m*数据量（需要数据分布，用来计算数据量）
# （3）设定遗忘因子
# （4）根据遗忘因子，预估质量（用来选择客户端）

输入：
（1）选中客户端集合：task_sleected_client：{0:[xx,xx,xx,xxx,x,],1:[xx,xx,xx,xxx,x,],2:[xx,xx,xx,xxx,x,]}
例：defaultdict(<class 'list'>, {0: [39, 22, 33, 15, 7, 40, 34, 32], 1: [16, 44, 1, 3, 23, 24, 0, 13, 43, 2, 11], 2: [47, 41, 29, 10, 30, 6, 36, 8, 12, 4]})

(2)开始时的测试损失：loss_start_task[list]:代表三个任务的损失 {0: [1, 1, 1], 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], 4: [0, 0, 0], 5: [0, 0, 0], 6: [0, 0, 0], 7: [0, 0, 0], 8: [0, 0, 0], 9: [0, 0, 0], 10: [0, 0, 0], 11: [0, 0, 0], 12: [0, 0, 0], 13: [0, 0, 0], 14: [0, 0, 0], 15: [0, 0, 0], 16: [0, 0, 0], 17: [0, 0, 0], 18: [0, 0, 0], 19: [0, 0, 0]})
例：defaultdict(<class 'list'>, {0: [1, 1, 1], 1: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 2: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 3: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 4: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 5: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 6: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 7: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 8: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 9: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 10: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 11: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 12: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 13: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 14: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 15: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 16: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 17: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 18: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225], 19: [2.2479846702575683, 2.2969743461608885, 3.1495694717407225]})

(3)结束时的训练损失：loss_end_task_client【list】
例如：[[0.3953128404376912, 0.23144705615361602, 0.31054921828291276, 0.6349611067702063, 0.42187520109646454, 0.41333054376053785, 0.4405809660236734, 0.3591128704510629], [0.5478954045809246, 1.0899911349666291, 1.9265997341700962, 1.5828337033589681, 0.6764125861751531, 0.6075855641777432, 0.9823405356953543, 0.8424297634314032, 1.1088735735693644, 0.39055607483942545, 0.8421094944798633], [1.6225390481948854, 23.46164998004311, 3.0986665785312653, 2.857364821434021, 1.1737479001283646, 80.74198991298675, 2.5638516227404273, 9.743762378692626, 449.49617836475375, 1.7164258718490601]]

（4）client_dict:三个任务的数据分布，用于计算总数据量[array]
(5)Quality_all:用于存放每一轮、所有任务、所有客户端的质量
（6）iter:轮数，用于改变Quality_all中的质量
# (5)遗忘因子：forget【一个值】

输出：
（1）客户端一轮针对一个任务的质量q（按照轮次的质量），[list]

步骤：
1、获取上一轮开始的测试损失、上一轮的客户端训练损失
"""

def FairGetQuality(task_sleected_client,loss_start_task, loss_end_task_client, client_dict, Quality_all, iter):

    # print("****************************************************************")
    # print("上一轮客户端选择结果：{}".format(task_sleected_client))
    # print("上一轮测试损失：{}".format(loss_start_task))
    # print("每个客户端的训练损失：{}".format(loss_end_task_client))
    # print("****************************************************************")

    #计算存放新一轮的质量
    Quality_task_client_iter = []

    #1、利用损失算m：开始时的测试损失-结束时的训练损失      三个任务，针对每个客户端{0:[],1:[],2:[]}
    #开始时的测试损失：一个确定的值
    #结束时的训练损失：一组客户端的（list）
    m_task_client = defaultdict(list)
    for j in range(len(task_sleected_client)):#3个任务
        for i in range(len(task_sleected_client[j])):#选中的每个客户端
            m = loss_start_task[iter-1][j] - loss_end_task_client[j][i]
            m_task_client[j].append(m)
    # print("计算出的m为：{}".format(m_task_client))

    # for i in range(len(loss_end_task_client)):
    #     m_task_client.append(loss_start_task - loss_end_task_client[i])

    #2、计算质量
    #(1)计算三个任务中每个客户端的总数据量
    sum_Data_client = list()
    for j in range(len(client_dict)):
        sum_Data_client_0 = list()
        # print("j = {}".format(j))
        for i in range(len(client_dict[j])):
            # print("i = {}".format(i))
            sum_Data_client_0.append(sum(client_dict[j][i]) / np.sum(client_dict[j]))

        sum_Data_client.append(sum_Data_client_0)
    # print("每个客户端的数据量之和为：{}".format(sum_Data_client))

    #(2)m*数据量
    for j in range(len(task_sleected_client)):
        for i in range(len(task_sleected_client[j])):
            m_Data = m_task_client[j][i] * sum_Data_client[j][task_sleected_client[j][i]]
            #不考虑数据量时
            # m_Data = m_task_client[j][i]
            Quality_all[iter-1][task_sleected_client[j][i]][j] = m_Data

    # print("归一化之前，客户端的质量为：{}".format(Quality_all))

    # 归一化
    # all_Quality = []
    # for iterm in range(len(Quality_all)):
    #     for i in range(len(Quality_all[iterm])):
    #         for j in range(len(Quality_all[iterm][i])):
    #             all_Quality.append(Quality_all[iterm][i][j])
    #
    # xmin = min(all_Quality)
    # xmax = max(all_Quality)
    #
    # for iterm in range(len(Quality_all)):
    #     for i in range(len(Quality_all[iterm])):
    #         for j in range(len(Quality_all[iterm][i])):
    #             Quality_all[iterm][i][j] = (Quality_all[iterm][i][j] - xmin) / (xmax - xmin)

    # print("归一化之后，客户端的质量为：{}".format(Quality_all))


    return Quality_all

# """
# *单个任务所选择的质量计算*
# FAIR方法的获取质量：
# （1）获取客户端训练的损失、测试损失，得到m
# （2）质量 = m*数据量（需要数据分布，用来计算数据量）
# # （3）设定遗忘因子
# # （4）根据遗忘因子，预估质量（用来选择客户端）
#
# 输入：
# （1）选中客户端集合：task_sleected_client[list]
# (2)开始时的测试损失：loss_start_task[list]:代表三个任务的损失
# (3)结束时的训练损失：loss_end_task_client【list】
# （4）client_dict_task:一个任务的数据分布，用于计算总数据量[array]
# (5)Quality_all:用于存放每一轮、所有任务、所有客户端的质量
# （6）iter:轮数，用于改变Quality_all中的质量
# # (5)遗忘因子：forget【一个值】
#
# 输出：
# （1）客户端一轮针对一个任务的质量q（按照轮次的质量），[list]
# """
# def FairGetQuality(task_sleected_client,loss_start_task, loss_end_task_client, client_dict_task, Quality_all, iter):
#
#     #计算新一轮的质量
#     Quality_task_client = []
#
#     #1、利用损失算m：开始时的测试损失-结束时的训练损失
#     #开始时的测试损失：一个确定的值
#     #结束时的训练损失：一组客户端的（list）
#     m_task_client = []
#     for i in range(len(loss_end_task_client)):
#         m_task_client.append(loss_start_task - loss_end_task_client[i])
#
#     #2、计算质量
#     sum_Data_client = []#用于存放50个客户端总的数据量
#     for i in range(len(client_dict_task)):
#         sum_Data_client.append(sum(client_dict_task[i]))
#
#
#     for i in range(len(task_sleected_client)):
#         Quality_task_client.append(m_task_client[i] * sum_Data_client[task_sleected_client[i]])
#
#     return Quality_task_client

'''
获取遗忘因子：与轮数相关

输入：
（1）遗忘因子值
(2)轮数


输出：
（1）遗忘因子列表[]

'''

def GetForgetFactor(forget_value, iter):
    forget_factor_value = []

    for i in range(iter):
        forget_factor_value.append(forget_value ** (iter - i))

    return forget_factor_value

'''
获取下一轮预估质量，用于CLQM筛选客户端

输入：
(1)每个客户端，每一轮，针对每个任务的质量Quality_All:[{0:[xx,xx,xx],1:[xx,xx,xx]...,49:[xx,xx,xx]},{0:[xx,xx,xx],1:[xx,xx,xx]...,49:[xx,xx,xx]},.......,{0:[xx,xx,xx],1:[xx,xx,xx]...,49:[xx,xx,xx]}]
(2)遗忘因子forget_factor_value：[]

输出：
(1)每个客户端的预估质量：Estimate_Quality:{0:[xx,xx,xx],1:[xx,xx,xx]...,49:[xx,xx,xx]}

'''

def GetEstimateQuality(Quality_All, forget_factor_value, iter, args):
    Estimate_Quality = defaultdict(list)

    #所有k轮，即只需要考虑前k次的Quality_All
    # iter_t = len(forget_factor_value)


    #计算预估质量
    for i in range(args.num_users):
        for j in range(len(Quality_All[iter][0])):
            sum_factor_quality = 0
            sum_factor = 0
            for iterm in range(len(forget_factor_value)):
                sum_factor_quality += forget_factor_value[iterm] * Quality_All[iterm][i][j]
                sum_factor += forget_factor_value[iterm]
            quality_estimate = sum_factor_quality / sum_factor
            Estimate_Quality[i].append(quality_estimate)
    # print("第{}轮的预估质量为：{}".format(iter, Estimate_Quality))


    # #第k轮，第i个客户端，第j个任务
    # for i in range(args.num_users):
    #     for j in range(len(Quality_All[0][0])):
    #         sum_factor_quality = 0
    #         sum_factor = 0
    #         for k in range(len(forget_factor_value)):
    #             sum_factor_quality += forget_factor_value[k] * Quality_All[k][i][j]
    #             sum_factor += forget_factor_value[k]
    #         quality_estimate = sum_factor_quality / sum_factor
    #         Estimate_Quality[i].append(quality_estimate)



    return Estimate_Quality








