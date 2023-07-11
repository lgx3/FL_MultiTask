'''
client_selection.py:对每个任务确定选择的客户端:
________________________________________________________________________________
函数1：getClientBudget(task_count, client_count)
作用：获取客户端报价

输入：
    task_count：（int）任务数量
    client_count：（int）客户端数量

输出：
    budget_dic：(dict(list)):客户端针对每个任务的报价
            例如： {0: [9.63294930345688, 2.0187577523713554], 1: [4.08323420006174, 3.5724288732536875],...,19: [4.69231785404729, 8.150881397207193]}

________________________________________________________________________________
函数2：clientSelect(Quality, Budget, Task, Task_Budget, X, X_demo)
作用：针对每一个任务（单个任务），找出满足条件的前k个客户端集合，不考虑选没选，只按排序找出前k个

输入：
    Quality：（dict（list））所有任务的质量
            例如：{0: [0.05432758596230657, 0.04531593186150271], 1: [0.045391643187727385, 0.06010152837008204],..., 19: [0.02662944314035432, 0.05933503960261935]}
    Budget：(dict(list)):客户端针对每个任务的报价
            例如： {0: [9.63294930345688, 2.0187577523713554], 1: [4.08323420006174, 3.5724288732536875],...,19: [4.69231785404729, 8.150881397207193]}
    Task：(str):单个task的名字
    Task_Budget：(list):每个任务的预算
            例如：Budget_Task = [100, 120]
    X：(list):CLQM算法的客户端分配情况，0为未分配，1为已分配
    X_demo:(list):贪心算法的客户端分配情况，0为未分配，1为已分配

输出：
    client_selected：(list):针对输入的任务Task，CLQM选中客户端列表
    k_smallest：（int）：CLQM找到的最小的k
    X_demo：(list):贪心算法的客户端分配情况，0为未分配，1为已分配

________________________________________________________________________________
函数3：TaskSelected(X, client_selected, dic_q, Task, Task_p)
作用：选择质量最大的任务，确定质量最大的任务的客户端

输入：
    X：(list):客户端的分配情况
    client_selected：（dict（list））选中的客户端
            例如：{0: [10, 7, 12, 17, 14, 3, 16, 18], 1: [15, 0, 12, 1, 7, 11, 10, 4]}
    dic_q：（dict（list））客户端针对每个任务的质量
            例如：{0: [0.05432758596230657, 0.04531593186150271], 1: [0.045391643187727385, 0.06010152837008204],...,19: [0.02662944314035432, 0.05933503960261935]}
    Task：（list）任务列表。例如：[0, 1]
    Task_p：（list）任务是否被分配。例如：[0, 0]

输出：
    Task_Quality.argsort()[-1]：(list):返回每个任务所选中客户端的总质量最大的下标

________________________________________________________________________________
函数4：ConfirmClientForTask(S, X, task, task_selected_cliented)
作用：为任务确定客户端，找到质量最大的任务之后，将其中客户端x置为1（未用到）

输入：
    S：（list）任务是否被分配。例如：[0, 0, 0,...,0]
    X：（list）任务是否可以被分配。例如：[1, 1, 1,...,1]
    task：（list）任务序号
    task_selected_cliented：（dict（list））选中的客户端
            例如：{0: [10, 7, 12, 17, 14, 3, 16, 18], 1: [15, 0, 12, 1, 7, 11, 10, 4]}

输出：
    X:同上
    S：同上

________________________________________________________________________________
函数5：CLQM(dic_q, budget, args)
作用：将所有函数组合，确定每个任务选中的客户端

输入：
    dic_q：（dict（list））客户端针对所有任务的质量
            例如：{0: [0.05432758596230657, 0.04531593186150271], 1: [0.045391643187727385, 0.06010152837008204],...,19: [0.02662944314035432, 0.05933503960261935]}
    budget：（dict（list））客户端针对每个任务的报价
            例如：{0: [9.63294930345688, 2.0187577523713554], 1: [4.08323420006174, 3.5724288732536875],...,19: [4.69231785404729, 8.150881397207193]}
    args：

输出：
    client_selected_every_task：（dict（list））每个任务选中的客户端，最终想得到的结果
            例如： {0: [10, 7, 12, 17, 14, 3, 16, 18], 1: [15, 0, 1, 11, 4]}


'''

import numpy as np
from collections import defaultdict
import random
from util.dirichlet import getAllClientDataDistribution, getTestDistribution

#获取报价
def getClientBudget(task_count, client_count, args):
    np.random.seed(args.seed)
    budget_dic = defaultdict(list)
    for i in range(client_count):
        a = np.random.uniform(1, 5, task_count)
        budget_dic[i] += list(a)
    return budget_dic

#根据质量生成报价
#client_quality例如：{0: [0.05432758596230657, 0.04531593186150271], 1: [0.045391643187727385, 0.06010152837008204],..., 19: [0.02662944314035432, 0.05933503960261935]}
'''
作用：根据客户端质量生成客户端报价

输入：
    task_count：（int）任务数量
    client_count：（int）客户端数量
    client_quality：客户端质量

输出：
    budget_dic：(dict(list)):客户端针对每个任务的报价
            例如： {0: [9.63294930345688, 2.0187577523713554], 1: [4.08323420006174, 3.5724288732536875],...,19: [4.69231785404729, 8.150881397207193]}

'''
def getClientBudget_q(client_quality, args):
    np.random.seed(args.seed)
    budget_dic = defaultdict(list)
    for i in range(len(client_quality)):
        for j in range(len(client_quality[i])):
            a = np.random.uniform(40*client_quality[i][j],60*client_quality[i][j],1)
            budget_dic[i].append(float(a))
    return budget_dic


#预算之内可以选择的客户端
'''
    输入：客户端针对每个任务的质量、客户端针对每个任务的报价、任务号、任务预算
    输出：选中客户端集合
'''
def clientSelect(Quality, Budget, Task, Task_Budget, X, X_demo):
    # q/b用于排序
    quality_per_budget = []
    # b/q用于筛选
    budget_per_quality = []
    for i in range(len(Quality)):
        # print(Quality[i][Task])
        # print(Budget[i][Task])
        quality_per_budget.append(Quality[i][Task] / Budget[i][Task])
        budget_per_quality.append(Budget[i][Task] / Quality[i][Task])

    # print("第{}个任务，所有人单位预算下的质量(用于排序)：{}".format(Task, quality_per_budget))
    # print("第{}个任务，所有人单位质量下的预算（用于筛选）：{}\n".format(Task, budget_per_quality) )

    client_sorted = np.argsort(quality_per_budget) #升序排序
    client_sorted = client_sorted[::-1] #转为降序（翻转了一下）
    # print('任务{}的客户端排序结果{}\n'.format(Task, client_sorted))
    # print("根据单位质量进行降序排序：", client_sorted)

    # print("→EMD-MQFL算法：")

    k_smallest = 0
    for k in range(len(client_sorted)):
        selected_clients_sum_payments = 0
        # for i in range(k+1):
        #     sum_budget_per_quality = budget_per_quality[client_sorted[i]]
        #
        # for i in range(k+1):
        #     selected_clients_sum_payments += sum_budget_per_quality * Quality[client_sorted[i]][Task] * X[client_sorted[i]]

        for i in range(k):
            selected_clients_sum_payments += budget_per_quality[client_sorted[k]] * Quality[client_sorted[i]][Task] * X[client_sorted[i]]

        if selected_clients_sum_payments > Task_Budget:
            k_smallest = k-1
            break

    # print('smallest k = {}'.format(k_smallest))

    client_selected = []
    client_selected_payment = []
    for i in range(k_smallest):
        # if X[client_sorted[i]] == 1:
        client_selected.append(client_sorted[i])
        #支付：
        client_selected_payment.append(budget_per_quality[client_sorted[k_smallest]] * Quality[client_sorted[i]][Task])
    # print("预选中客户端的支付价格为：{}".format(client_selected_payment))
    # print('预算内,EMD_MQFL算法选中的客户端集合：{}'.format(client_selected))


    return client_selected,client_selected_payment, k_smallest, X_demo


# 选质量最大的任务
def TaskSelected(X, client_selected, dic_q, Task, Task_p):
    # print("————被选中的客户端————：{}".format(client_selected))
    # print("此时X的情况：{}".format(X))
    # 所有任务的总质量
    Task_Quality = []
    for task in Task:
        # 某个任务的客户端总质量
        if Task_p[task] == 0:
            sum_quality = 0
            for client in client_selected[task]:
                if X[client] == 1:
                    sum_quality += dic_q[client][task]
            Task_Quality.append(sum_quality)
        else:
            Task_Quality.append(0)
    # print('任务的总质量：{}'.format(Task_Quality))

    Task_Quality = np.array(Task_Quality)
    return Task_Quality.argsort()[-1]




def ConfirmClientForTask(S, X, task, task_selected_cliented):
    for client in task_selected_cliented:
        if X[client] == 1:
            S[client][task] = 1
            X[client] = 0
    return X, S


#确定客户端、每个客户端支付的价格
def CLQM(dic_q, budget, Budget_Task, args):

    # 任务列表
    Task = [0, 1, 2]
    Task1 = [0, 1, 2]
    # 任务分配标志，0：未分配,1：已分配
    Task_p = [0] * 3
    # 每个任务分配的客户端
    client_selected_every_task = defaultdict(list)
    #未确定的客户端支付的报酬，中间过渡
    client_selected_every_task_payment_0 = defaultdict(list)
    client_selected_every_task_payment = defaultdict(list)

    #最终确定的任务
    task_selected_finally = 0

    # 客户端分配标志(自己的算法), 0：不可分配, 1：可分配
    X = [1] * args.num_users
    # 客户端分配标志(贪心算法), 0：不可分配, 1：可分配
    X_demo = [1] * args.num_users
    # 客户端参加哪个任务
    S = defaultdict(list)
    for i in range(50):
    # for i in range(args.num_users):
        S[i] += [0, 0, 0]

    count = 1
    while sum(X) > 0 and sum(Task_p) < 3:
        # print('\n———————————————————————— 第{}次计算：针对每个任务，找出满足条件限制的前k个客户端 ————————————————————————'.format(count))
        # 每个任务可一参加的客户端
        K_smallest = {}
        client_selected_AllTask = defaultdict(list)
        count += 1
        for task in Task:
            # print('\n~~~~~~~~~~   考虑第{}个任务时   ~~~~~~~~~~'.format(task))
            client_selected, client_payment, k_smallest, X_demo = clientSelect(dic_q, budget, task, Budget_Task[task], X, X_demo)
            client_selected_every_task_payment_0[task] = client_payment

            K_smallest.update({task: k_smallest})
            client_selected_AllTask[task] += client_selected

        # 找总质量最大的任务
        task_selected = TaskSelected(X, client_selected_AllTask, dic_q, Task1, Task_p)

        # print('选中的质量最大的任务是{}'.format(task_selected))
        Task_p[task_selected] = 1
        Task.remove(task_selected)
        # print('任务分配结果{}'.format(Task_p))

        client_selected = []
        # print('任务{}预选中的客户端{}'.format(task, client_selected_AllTask[task_selected]))
        # print(K_smallest[task_selected])
        for i in range(K_smallest[task_selected]):
            client = client_selected_AllTask[task_selected][i]
            if X[client] == 1:
                client_selected.append(client)
                client_selected_every_task_payment[task_selected].append(client_selected_every_task_payment_0[task_selected][i])
                X[client] = 0
        # print("选中客户端集合:", client_selected)
        client_selected_every_task[task_selected] += client_selected

    return client_selected_every_task, client_selected_every_task_payment

#随机选择客户端，确定客户端、每个客户端支付的价格
def CLQM_Random(dic_q, budget, Budget_Task, args):

    # 任务列表
    Task = [0, 1, 2]
    Task_for_random = [0, 1, 2] # 用于随机选取任务的任务集合
    # 任务分配标志，0：未分配,1：已分配
    Task_p = [0] * 3
    # 每个任务分配的客户端
    client_selected_every_task = defaultdict(list)
    #未确定的客户端支付的报酬，中间过渡
    client_selected_every_task_payment_0 = defaultdict(list)
    client_selected_every_task_payment = defaultdict(list)

    #最终确定的任务
    task_selected_finally = 0

    # 客户端分配标志(自己的算法), 0：不可分配, 1：可分配
    X = [1] * args.num_users
    # 客户端分配标志(贪心算法), 0：不可分配, 1：可分配
    X_demo = [1] * args.num_users
    # 客户端参加哪个任务
    S = defaultdict(list)
    for i in range(50):
    # for i in range(args.num_users):
        S[i] += [0, 0, 0]

    count = 1
    while sum(X) > 0 and sum(Task_p) < 3:
        # print('\n———————————————————————— 第{}次计算：针对每个任务，找出满足条件限制的前k个客户端 ————————————————————————'.format(count))
        # 每个任务可一参加的客户端
        K_smallest = {}
        client_selected_AllTask = defaultdict(list)
        count += 1
        for task in Task:
            # print('\n~~~~~~~~~~   考虑第{}个任务时   ~~~~~~~~~~'.format(task))
            client_selected, client_payment, k_smallest, X_demo = clientSelect(dic_q, budget, task, Budget_Task[task], X, X_demo)
            client_selected_every_task_payment_0[task] = client_payment

            K_smallest.update({task: k_smallest})
            client_selected_AllTask[task] += client_selected


        # 找总质量最大的任务
        task_selected = np.random.choice(Task, 1)[0]
        Task.remove(task_selected)

        Task_p[task_selected] = 1

        # task_selected = TaskSelected(X, client_selected_AllTask, dic_q, Task1, Task_p)
        #
        # # print('选中的质量最大的任务是{}'.format(task_selected))
        #
        # Task.remove(task_selected)
        # # print('任务分配结果{}'.format(Task_p))



        client_selected = []
        # print('任务{}预选中的客户端{}'.format(task, client_selected_AllTask[task_selected]))
        # print(K_smallest[task_selected])
        for i in range(K_smallest[task_selected]):
            client = client_selected_AllTask[task_selected][i]
            if X[client] == 1:
                client_selected.append(client)
                client_selected_every_task_payment[task_selected].append(client_selected_every_task_payment_0[task_selected][i])
                X[client] = 0
        # print("选中客户端集合:", client_selected)
        client_selected_every_task[task_selected] += client_selected

    return client_selected_every_task, client_selected_every_task_payment



# '''
# EMD_MQFL
# 输入：报价、预算、任务、质量
#
#
# （1）先对每个任务的质量进行排序：clientSelect（），选出一批客户端
# （2）
# '''
# def Emd_MQFL(Budget, Budget_Task, Task, dic_q):
#     Emd_MQFL_Client_Selected = defaultdict(list)
#     Emd_MQFL_Client_Selected_payments = defaultdict(list)
#
#     #用于中间过渡
#     Emd_MQFL_Client_Selected_Undetermined = defaultdict(list)
#     Emd_MQFL_Client_Selected_payments_Undetermined = defaultdict(list)
#
#     #用于排序
#     Quality_Per_Budget = defaultdict(list)
#     #用于计算筛选
#     Budget_Per_Quality = defaultdict(list)
#
#     X = [1]*50
#
#     #（1）计算q/b
#     for i in range(len(Budget)):
#         for j in range(len(Budget[i])):
#             Quality_Per_Budget[j].append(dic_q[i][j] / Budget[i][j])
#             Budget_Per_Quality[j].append(Budget[i][j] / dic_q[i][j])
#
#     print("q/b = {}".format(Quality_Per_Budget))
#
#     #（2）对每个任务进行排序，并返回排序之前所对应的客户端
#     Quality_Per_Budget_sorted = defaultdict(list)
#     Quality_Per_Budget_sorted_id = defaultdict(list)
#     for i in range(len(Quality_Per_Budget)):
#         Quality_Per_Budget_sorted[i] += sorted(Quality_Per_Budget[i],reverse=True)
#         Quality_Per_Budget_sorted_id[i] += sorted(range(len(Quality_Per_Budget[i])), key=lambda x: Quality_Per_Budget[i][x], reverse=True)
#
#     print("排序之后的q/b为：{}".format(Quality_Per_Budget_sorted))
#     print("排序之后的q/b，对应的客户端为：{}".format(Quality_Per_Budget_sorted_id))
#
#     #(3)排序完成，找到每个任务的关键k，选出前k个客户端
#     k_smallest = defaultdict(list)
#     for i in range(len(Quality_Per_Budget_sorted)):
#         # print("第{}个任务".format(i))
#         for j in range(len(Quality_Per_Budget_sorted[i])):
#             # print("第{}个客户端".format(j))
#             sum_Budget_Per_Quality = 0
#             if j == 0:
#                 sum_Budget_Per_Quality = Budget_Per_Quality[i][Quality_Per_Budget_sorted_id[i][j]]
#             if j != 0:
#                 for z in range(j+1):
#                     sum_Budget_Per_Quality += Budget_Per_Quality[i][Quality_Per_Budget_sorted_id[i][j]]
#
#             if sum_Budget_Per_Quality * dic_q[j][i] * X[j] > Budget_Task[i]:
#                 k_smallest[i] = j
#                 break
#     print("k的值为：{}".format(k_smallest))
#
#     for i in range(len(k_smallest)):
#         for j in range(k_smallest[i]):
#             Emd_MQFL_Client_Selected_Undetermined[i].append(Quality_Per_Budget_sorted_id[i][j])
#             Emd_MQFL_Client_Selected_payments_Undetermined[i].append(Budget_Per_Quality[i][Quality_Per_Budget_sorted_id[i][k_smallest]] * dic_q[Quality_Per_Budget_sorted_id[i][j]][i])
#
#     #(4)找出质量最大的任务，一个任务一个任务的计算
#     Task_all_q
#     for i in range(len(Emd_MQFL_Client_Selected_Undetermined)):
#         for j in range
#
#
#
#     return Emd_MQFL_Client_Selected, Emd_MQFL_Client_Selected_payments




# def Payments_All(client_selected_every_task,budget_dic):
#     Task_All_Payments = [0, 0]
#
#     for i in range(2):
#         for j in range(len(client_selected_every_task[i])):
#             Task_All_Payments[i] += budget_dic[client_selected_every_task[i][j]][i]
#
#         # print('任务{}按照报价支付时，支付总额为:{}'.format(i,Task_All_Payments[i]))
#
#     return Task_All_Payments

# 三个任务预算时的报价优先，支付报价
def bidPriceFirst_ClientSelection(Budget, Budget_Task):
    #Budget样式：{0: [9.63294930345688, 2.0187577523713554], 1: [4.08323420006174, 3.5724288732536875],...,19: [4.69231785404729, 8.150881397207193]}
    client_selected = defaultdict(list)
    # Task = [0, 1]
    Bidfirst_payment = defaultdict(list)
    All_payment = [0, 0, 0]

    #1、所有客户端按照报价排序
    All_Client_Budget = []
    for i in range(len(Budget)):
        for j in range(len(Budget[i])):
            All_Client_Budget.append(Budget[i][j])

    Sort_All_Budget = sorted(All_Client_Budget)

    #2、选客户端
    s = [0]*len(Budget)#判断客户端是否被选中：0为未被选过，1为已经被选

    for i in range(len(Sort_All_Budget)):
        x = Sort_All_Budget[i]
        for j in range(len(Budget)):
            for z in range(len(Budget[j])):
                if x == Budget[j][z] and s[j] == 0 and All_payment[z] + x <= Budget_Task[z]:
                    client_selected[z].append(j)
                    All_payment[z] += Budget[j][z]
                    s[j] = 1

    for i in range(len(client_selected)):
        for j in range(len(client_selected[i])):
            Bidfirst_payment[i].append(Budget[client_selected[i][j]][i])

    return client_selected, Bidfirst_payment

'''
每个任务有自己单独的预算，背包贪婪选择客户端
'''
def Individual_Knapsack_greedy(Budget, client_dict, Budget_Task):
    Knapsack_greedy_client_selected = defaultdict(list)
    Knapsack_greedy_payment = defaultdict(list)

    client_all_data = defaultdict(list)
    # （1）计算每个客户端每个任务的总数据量
    for i in range(len(client_dict)):
        for j in range(len(client_dict[i])):
            client_all_data[j].append(sum(client_dict[i][j]))

    # print("客户端每个任务的数据量为：{}".format(client_all_data))

    # （2）获取单位数据量的成本
    Budget_per_Data = defaultdict(list)
    for i in range(len(Budget)):
        for j in range(len(Budget[i])):
            Budget_per_Data[i].append(Budget[i][j] / client_all_data[i][j])

    # print("单位质量的成本为：{}".format(Budget_per_Data))

    # （3）每个客户端，找最便宜的任务
    min_Budget_per_Data = []
    min_Budget_per_Data_Task = []
    for i in range(len(Budget_per_Data)):
        min_Budget_per_Data.append(min(Budget_per_Data[i]))
        for j in range(len(Budget_per_Data[i])):
            if min(Budget_per_Data[i]) == Budget_per_Data[i][j]:
                min_Budget_per_Data_Task.append(j)

    # print("背包贪婪客户端，单位数据成本为：{}".format(min_Budget_per_Data))
    # print("背包贪婪客户端，单位数据成本对应任务为：{}".format(min_Budget_per_Data_Task))

    # (4)对找到的每个客户端的最便宜的任务，进行总排序
    min_Budget_per_Data_sorted = sorted(min_Budget_per_Data)
    min_Budget_per_Data_sorted_id = sorted(range(len(min_Budget_per_Data)), key=lambda x: min_Budget_per_Data[x])

    # print("背包贪婪客户端，最小的单位数据成本再次升序排序为：{}".format(min_Budget_per_Data_sorted))
    # print("背包贪婪客户端，最小的单位数据成本排序后对应原来的客户端为：{}".format(min_Budget_per_Data_sorted_id))

    # (5)贪婪的选择客户端，支付报价(三个任务单独考虑)
    sum_allpayment = [0, 0, 0]
    for client in min_Budget_per_Data_sorted_id:
        for j in range(len(Budget)):
            if min_Budget_per_Data_Task[client] == j and (sum_allpayment[j] + Budget[client][min_Budget_per_Data_Task[client]]) <= Budget_Task[j]:
                Knapsack_greedy_client_selected[j].append(client)
                Knapsack_greedy_payment[j].append(Budget[client][min_Budget_per_Data_Task[client]])
                sum_allpayment[j] += Budget[client][min_Budget_per_Data_Task[client]]

    # 检查：
    # sum_all = 0
    # client_num = 0
    # for i in range(len(Knapsack_greedy_payment)):
    #     sum_all += sum(Knapsack_greedy_payment[i])
    #     client_num += len(Knapsack_greedy_payment[i])
    #
    # print("报价优先总支付：{}".format(sum_all))
    # print("选中总客户端：{}".format(client_num))

    return Knapsack_greedy_client_selected, Knapsack_greedy_payment




'''
随机选择客户端进行训练
输入：客户端报价、任务总预算、任务、args

#Budget样式：{0: [9.63294930345688, 2.0187577523713554], 1: [4.08323420006174, 3.5724288732536875],...,19: [4.69231785404729, 8.150881397207193]}
X样式：[14, 8, 5, 10, 1, 2, 17, 0, 7, 13, 15, 16, 19, 3, 18, 12, 11, 9, 4, 6]

'''
def RandomClientSelect(Budget, Budget_Task, args):
    random.seed(args.seed)
    client_selected = defaultdict(list)
    Random_payment = defaultdict(list)
    sum_payment = [0, 0, 0]

    #1、随机生成数据列表

    X = random.sample(range(0,args.num_users),args.num_users)
    print("随机生成的随机数为：{}".format(X))
    #
    # X = []
    # for i in range(args.num_users):
    #     X.append(i)

    # 2、不超预算，按照单双数进行选择
    for i in range(len(X)):
        if i % 3 == 0 and (sum_payment[0] + Budget[X[i]][0]) <= Budget_Task[0]:
            client_selected[0].append(X[i])
            sum_payment[0] += Budget[X[i]][0]
        elif i % 3 == 1 and (sum_payment[1] + Budget[X[i]][1]) <= Budget_Task[1]:
            client_selected[1].append(X[i])
            sum_payment[1] += Budget[X[i]][1]
        elif i % 3 == 2 and (sum_payment[2] + Budget[X[i]][2]) <= Budget_Task[2]:
            client_selected[2].append(X[i])
            sum_payment[2] += Budget[X[i]][2]

    # for i in range(len(X)):
    #     if i % 3 == 0 and (sum_payment[2] + Budget[X[i]][2]) <= Budget_Task[2]:
    #         client_selected[2].append(X[i])
    #         sum_payment[2] += Budget[X[i]][2]
    #     elif i % 3 == 1 and (sum_payment[0] + Budget[X[i]][0]) <= Budget_Task[0]:
    #         client_selected[0].append(X[i])
    #         sum_payment[0] += Budget[X[i]][0]
    #     elif i % 3 == 2 and (sum_payment[1] + Budget[X[i]][1]) <= Budget_Task[1]:
    #         client_selected[1].append(X[i])
    #         sum_payment[1] += Budget[X[i]][1]

    for i in range(len(client_selected)):
        for j in range(len(client_selected[i])):
            Random_payment[i].append(Budget[client_selected[i][j]][i])

    return client_selected, Random_payment



'''------------------------------------------------上面为任务单独预算的情况------------------------------------------------------'''
'''------------------------------------------------下面为总预算的情况------------------------------------------------------'''



'''
保证诚实性：
报价优先选择客户端：所有任务所有报价一起排序，选择报价低的客户端
输入：报价，任务预算
'''
# def bidPriceFirst_ClientSelection(Budget, Budget_Task):
#     client_selected = defaultdict(list)
#     Bidfirst_payment = defaultdict(list)
#
#     #存放客户端报价较低的任务的报价
#     min_Client_Budget = []
#     # 存放客户端报价较低的任务
#     min_Client_Budget_Task = []
#     #筛选出来的每个客户端最小的报价，再排序
#     # min_Client_Budget_sorted = []
#     #对应排序之前的位置，即客户端
#     # min_Client_Budget_sorted_id = []
#
#     #(1)先按报价排序
#     for i in range(len(Budget)):
#         min_Client_Budget.append(min(Budget[i]))
#         for j in range(len(Budget[i])):
#             if min(Budget[i]) == Budget[i][j]:
#                 min_Client_Budget_Task.append(j)
#
#     #(2)对筛选出来的客户端排序
#     min_Client_Budget_sorted = sorted(min_Client_Budget)
#     min_Client_Budget_sorted_id = sorted(range(len(min_Client_Budget)), key=lambda x: min_Client_Budget[x])
#
#     # print("报价优先筛选出来的客户端报价为：{}".format(min_Client_Budget))
#     # print("报价优先筛选出来的客户端，对应的任务为：{}".format(min_Client_Budget_Task))
#     # print("报价优先筛选出来的客户端排序后的报价为：{}".format(min_Client_Budget_sorted))
#     # print("报价优先筛选出来的客户端，排序后对应原来的索引为：{}".format(min_Client_Budget_sorted_id))
#
#     #筛选，找出关键k，选前k个客户端
#     k_smallest = 0
#     for i in range(len(min_Client_Budget_sorted)):
#         if i == 0:
#             continue
#         if min_Client_Budget_sorted[i] > Budget_Task / (i+1):
#             k_smallest = i
#             break
#     # print("k_smallest = {}".format(k_smallest))
#
#     client_selected_Undetermined = []
#     for i in range(k_smallest):
#         client_selected_Undetermined.append(min_Client_Budget_sorted_id[i])
#
#     # print("报价优先选择了{}个客户端，为:{}".format(len(client_selected_Undetermined), client_selected_Undetermined))
#
#     # avg_payment = Budget_Task / (k_smallest-1)
#     payment = min(Budget_Task / (k_smallest), min_Client_Budget_sorted[k_smallest])
#
#     for i in range(len(client_selected_Undetermined)):
#         if min_Client_Budget_Task[client_selected_Undetermined[i]] == 0:
#             client_selected[0].append(client_selected_Undetermined[i])
#             Bidfirst_payment[0].append(payment)
#
#         if min_Client_Budget_Task[client_selected_Undetermined[i]] == 1:
#             client_selected[1].append(client_selected_Undetermined[i])
#             Bidfirst_payment[1].append(payment)
#
#         if min_Client_Budget_Task[client_selected_Undetermined[i]] == 2:
#             client_selected[2].append(client_selected_Undetermined[i])
#             Bidfirst_payment[2].append(payment)
#
#
#     #检查：
#     # sum_all = 0
#     # for i in range(len(Bidfirst_payment)):
#     #     sum_all += sum(Bidfirst_payment[i])
#     #
#     # print("报价优先总支付：{}".format(sum_all))
#
#
#     return client_selected, Bidfirst_payment






#报价优先选择客户端：所有任务所有报价一起排序，选择报价低的客户端
'''
不保证诚实性：支付报价
报价优先选择客户端：所有任务所有报价一起排序，选择报价低的客户端
输入：报价，任务预算
Budget样式：{0: [9.63294930345688, 2.0187577523713554], 1: [4.08323420006174, 3.5724288732536875],...,19: [4.69231785404729, 8.150881397207193]}
'''
def bidPriceFirst_ClientSelection_all_Budget(Budget, Budget_Task):
    Bidfirst_all_Budget_client_selected = defaultdict(list)
    Bidfirst_all_Budget_payment = defaultdict(list)

    All_payment = 0

    # 存放客户端报价较低的任务的报价
    min_Client_Budget = []
    # 存放客户端报价较低的任务
    min_Client_Budget_Task = []
    # 筛选出来的每个客户端最小的报价，再排序
    # min_Client_Budget_sorted = []
    # 对应排序之前的位置，即客户端
    # min_Client_Budget_sorted_id = []

    # (1)先按报价排序
    for i in range(len(Budget)):
        min_Client_Budget.append(min(Budget[i]))
        for j in range(len(Budget[i])):
            if min(Budget[i]) == Budget[i][j]:
                min_Client_Budget_Task.append(j)

    # (2)对筛选出来的客户端再排序，然后返回对应的任务
    min_Client_Budget_sorted = sorted(min_Client_Budget)
    min_Client_Budget_sorted_id = sorted(range(len(min_Client_Budget)), key=lambda x: min_Client_Budget[x])


    #（3）选择客户端
    All_payment = 0
    #选中但未分配任务的客户端
    client_selected_Undetermined = []
    for i in range(len(min_Client_Budget_sorted)):
        if (All_payment + min_Client_Budget_sorted[i]) <= Budget_Task:
            client_selected_Undetermined.append(min_Client_Budget_sorted_id[i])
            All_payment += min_Client_Budget_sorted[i]

    #(4)给任务分配客户端
    for i in range(len(client_selected_Undetermined)):
        for j in range(len(Budget[i])):
            if min_Client_Budget_Task[client_selected_Undetermined[i]] == j:
                Bidfirst_all_Budget_client_selected[j].append(client_selected_Undetermined[i])
                Bidfirst_all_Budget_payment[j].append(Budget[client_selected_Undetermined[i]][j])

    # 检查：
    sum_all = 0
    for i in range(len(Bidfirst_all_Budget_payment)):
        sum_all += sum(Bidfirst_all_Budget_payment[i])
    print("报价优先总支付：{}".format(sum_all))

    #选中客户端
    client_num = len(Bidfirst_all_Budget_client_selected[0]) + len(Bidfirst_all_Budget_client_selected[1]) + len(Bidfirst_all_Budget_client_selected[2])
    print("选中客户端数量：{}".format(client_num))

    return Bidfirst_all_Budget_client_selected, Bidfirst_all_Budget_payment



'''
总预算情况下：不诚实的背包贪婪算法选择客户端：只考虑数据量和报价，支付报价，不考虑诚实性
'''
def Knapsack_greedy(Budget, client_dict, Budget_Task):
    Knapsack_greedy_client_selected = defaultdict(list)
    Knapsack_greedy_payment = defaultdict(list)

    client_all_data = defaultdict(list)
    #（1）计算每个客户端每个任务的总数据量
    for i in range(len(client_dict)):
        for j in range(len(client_dict[i])):
            client_all_data[j].append(sum(client_dict[i][j]))

    # print("客户端每个任务的数据量为：{}".format(client_all_data))

    #（2）获取单位数据量的成本
    Budget_per_Data = defaultdict(list)
    for i in range(len(Budget)):
        for j in range(len(Budget[i])):
            Budget_per_Data[i].append(Budget[i][j] / client_all_data[i][j])

    # print("单位质量的成本为：{}".format(Budget_per_Data))

    #（3）每个客户端，找最便宜的任务
    min_Budget_per_Data = []
    min_Budget_per_Data_Task = []
    for i in range(len(Budget_per_Data)):
        min_Budget_per_Data.append(min(Budget_per_Data[i]))
        for j in range(len(Budget_per_Data[i])):
            if min(Budget_per_Data[i]) == Budget_per_Data[i][j]:
                min_Budget_per_Data_Task.append(j)

    # print("背包贪婪客户端，单位数据成本为：{}".format(min_Budget_per_Data))
    # print("背包贪婪客户端，单位数据成本对应任务为：{}".format(min_Budget_per_Data_Task))

    #(4)对找到的每个客户端的最便宜的任务，进行总排序
    min_Budget_per_Data_sorted = sorted(min_Budget_per_Data)
    min_Budget_per_Data_sorted_id = sorted(range(len(min_Budget_per_Data)), key=lambda x: min_Budget_per_Data[x])

    # print("背包贪婪客户端，最小的单位数据成本再次升序排序为：{}".format(min_Budget_per_Data_sorted))
    # print("背包贪婪客户端，最小的单位数据成本排序后对应原来的客户端为：{}".format(min_Budget_per_Data_sorted_id))


    #(5)贪婪的选择客户端，支付报价
    sum_allpayment = 0
    for client in min_Budget_per_Data_sorted_id:
        for j in range(len(Budget)):
            if min_Budget_per_Data_Task[client] == j and (sum_allpayment + Budget[client][min_Budget_per_Data_Task[client]]) <= Budget_Task:
                Knapsack_greedy_client_selected[j].append(client)
                Knapsack_greedy_payment[j].append(Budget[client][min_Budget_per_Data_Task[client]])
                sum_allpayment += Budget[client][min_Budget_per_Data_Task[client]]

    # 检查：
    # sum_all = 0
    # for i in range(len(Knapsack_greedy_payment)):
    #     sum_all += sum(Knapsack_greedy_payment[i])
    #
    # print("报价优先总支付：{}".format(sum_all))


    return Knapsack_greedy_client_selected, Knapsack_greedy_payment


'''
总预算情况下：诚实的背包贪婪算法选择客户端：只考虑数据量和报价，支付关键价格，考虑诚实性的时候
'''
def Knapsack_greedy_truthful(Budget, client_dict, Budget_Task):
    Knapsack_greedy_client_selected = defaultdict(list)
    Knapsack_greedy_payment = defaultdict(list)
    client_selected_Undetermined = []

    client_all_data = defaultdict(list)
    #（1）计算每个客户端每个任务的总数据量
    for i in range(len(client_dict)):
        for j in range(len(client_dict[i])):
            client_all_data[j].append(sum(client_dict[i][j]))

    # print("客户端每个任务的数据量为：{}".format(client_all_data))


    #（2）获取单位数据量的成本
    Budget_per_Data = defaultdict(list)
    for i in range(len(Budget)):
        for j in range(len(Budget[i])):
            Budget_per_Data[i].append(Budget[i][j] / client_all_data[i][j])

    # print("单位质量的成本为：{}".format(Budget_per_Data))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    #（3）每个客户端，找最便宜的任务
    min_Budget_per_Data = []
    min_Budget_per_Data_Task = []
    for i in range(len(Budget_per_Data)):
        min_Budget_per_Data.append(min(Budget_per_Data[i]))
        for j in range(len(Budget_per_Data[i])):
            if min(Budget_per_Data[i]) == Budget_per_Data[i][j]:
                min_Budget_per_Data_Task.append(j)

    # print("背包贪婪客户端，单位数据成本为：{}".format(min_Budget_per_Data))
    # print("背包贪婪客户端，单位数据成本对应任务为：{}".format(min_Budget_per_Data_Task))

    #(4)对找到的每个客户端的最便宜的任务，进行总排序
    min_Budget_per_Data_sorted = sorted(min_Budget_per_Data)
    min_Budget_per_Data_sorted_id = sorted(range(len(min_Budget_per_Data)), key=lambda x: min_Budget_per_Data[x])

    # print("背包贪婪客户端，最小的单位数据成本再次升序排序为：{}".format(min_Budget_per_Data_sorted))
    # print("背包贪婪客户端，最小的单位数据成本排序后对应原来的客户端为：{}".format(min_Budget_per_Data_sorted_id))


    #(5)贪婪的选择客户端，关键支付
    k_smallest = 0
    # Budget_Task_per_sum_d = 0
    for i in range(len(min_Budget_per_Data_sorted)):
        sum_d = 0
        if i == 0:
            sum_d = client_all_data[min_Budget_per_Data_sorted_id[i]][min_Budget_per_Data_Task[min_Budget_per_Data_sorted_id[i]]]
        if i != 0:
            for j in range(i + 1):
                sum_d += client_all_data[min_Budget_per_Data_sorted_id[i]][min_Budget_per_Data_Task[min_Budget_per_Data_sorted_id[i]]]

        if min_Budget_per_Data_sorted[i] > Budget_Task / sum_d:
            k_smallest = i
            Budget_Task_per_sum_d = Budget_Task / (sum_d - client_all_data[min_Budget_per_Data_sorted_id[k_smallest]][min_Budget_per_Data_Task[min_Budget_per_Data_sorted_id[k_smallest]]])
            k_smallest_Budget_per_sum_d = Budget[min_Budget_per_Data_sorted_id[k_smallest]][min_Budget_per_Data_Task[min_Budget_per_Data_sorted_id[k_smallest]]] / client_all_data[min_Budget_per_Data_sorted_id[k_smallest]][min_Budget_per_Data_Task[min_Budget_per_Data_sorted_id[k_smallest]]]
            # print("------k的值为：{}".format(k_smallest))
            # print("------平均单位质量的成本值为：{}".format(Budget_Task_per_sum_d))
            # print("------第k位客户端的平均单位质量的成本值为：{}".format(k_smallest_Budget_per_sum_d))
            break

    #（6）选择前k个客户端
    for i in range(k_smallest):
        client_selected_Undetermined.append(min_Budget_per_Data_sorted_id[i])

    min_keypayment = min(Budget_Task_per_sum_d, k_smallest_Budget_per_sum_d)

    #(7)给各个客户端分配任务
    for i in range(len(client_selected_Undetermined)):
        for j in range(len(Budget[i])):
            if min_Budget_per_Data_Task[client_selected_Undetermined[i]] == j:
                Knapsack_greedy_client_selected[j].append(client_selected_Undetermined[i])
                Knapsack_greedy_payment[j].append(client_all_data[client_selected_Undetermined[i]][j] * min_keypayment)

    # 检查：
    # sum_all = 0
    # for i in range(len(Knapsack_greedy_payment)):
    #     sum_all += sum(Knapsack_greedy_payment[i])

    # print("报价优先总支付：{}".format(sum_all))


    return Knapsack_greedy_client_selected, Knapsack_greedy_payment

'''
按照质量，一个客户端一个客户端的排序选择
输入：客户端数据质量dic_q
    客户端报价Budget
    任务预算Budget_Task
    
输出：选中客户端client_dict
'''
def maxQ(dic_q,Budget,Budget_Task):
    client_dict = defaultdict(list)
    dic_q_Per_Budget = defaultdict(list)
    sum_payment = [0, 0, 0]

    for i in range(len(dic_q)):
        for j in range(len(dic_q[i])):
            dic_q_Per_Budget[i].append(dic_q[i][j]/Budget[i][j])

    # print("单位成本质量为:{}".format(dic_q_Per_Budget))

    for i in range(len(dic_q_Per_Budget)):
        max_client = max(dic_q_Per_Budget[i])
        # print(max_client)
        for j in range(len(dic_q_Per_Budget[i])):
            if max_client ==dic_q_Per_Budget[i][j] and (sum_payment[j] + Budget[i][j]) <= Budget_Task[j]:
                client_dict[j].append(i)
                sum_payment[j] += Budget[i][j]

    return client_dict

'''
函数：在保证诚实性的同时，按照质量排序选择客户端
输入：客户端数据质量dic_q
    客户端报价Budget
    任务预算Budget_Task(int；类型的总预算)
输出：

'''
def maxQTruthfulness(dic_q,Budget,Budget_Task):
    client_dict = defaultdict(list)
    maxQTruthfulness_payment = defaultdict(list)
    #客户端原来的报价，用于检查客户端报酬是不是非负
    client_budget = defaultdict(list)

    #选过的客户端x为1，未选的为0
    x = [0]*50
    min_Budget_Per_dic_q = []
    min_Budget_Per_dic_q_task = []
    dic_q_Per_Budget = defaultdict(list)
    Budget_Per_dic_q = defaultdict(list)
    client_selected_Undetermined = []

    #1、先计算Budget_Per_dic_q，单位质量的成本，升序排序，越小越好
    for i in range(len(Budget)):
        for j in range(len(Budget[i])):
            Budget_Per_dic_q[i].append(Budget[i][j]/dic_q[i][j])
    # print(Budget_Per_dic_q)

    #2、再排序，选出每个客户端单位成本最小的任务
    for i in range(len(Budget_Per_dic_q)):
        min_Budget_Per_dic_q.append(min(Budget_Per_dic_q[i]))
        for j in range(len(Budget_Per_dic_q[i])):
            if min(Budget_Per_dic_q[i]) == Budget_Per_dic_q[i][j]:
                min_Budget_Per_dic_q_task.append(j)

    # print("未排序的客户端单位成本质量为：{}".format(min_Budget_Per_dic_q))
    # print("未排序的客户端单位成本质量对应的任务为：{}".format(min_Budget_Per_dic_q_task))
    # print(len(min_Budget_Per_dic_q_task))

    min_Budget_Per_dic_q_sorted = sorted(min_Budget_Per_dic_q)
    # min_Budget_Per_dic_q_sorted_id = sorted(range(len(min_Budget_Per_dic_q)), key=lambda x: min_Budget_Per_dic_q[x], reverse=True)
    min_Budget_Per_dic_q_sorted_id = sorted(range(len(min_Budget_Per_dic_q)), key=lambda x: min_Budget_Per_dic_q[x])
    # print("排序之后的客户端单位质量成本为：{}".format(min_Budget_Per_dic_q_sorted))
    # print("排序之后的客户端单位质量成本对应的客户端为：{}".format(min_Budget_Per_dic_q_sorted_id))


    #3、再筛选
    k_smallest = 0
    Budget_Task_per_sum_q = 0
    for i in range(len(min_Budget_Per_dic_q_sorted)):
        sum_q = 0
        if i ==0:
            sum_q = dic_q[min_Budget_Per_dic_q_sorted_id[i]][min_Budget_Per_dic_q_task[min_Budget_Per_dic_q_sorted_id[i]]]
        if i != 0 :
            for j in range(i+1):
                sum_q += dic_q[min_Budget_Per_dic_q_sorted_id[j]][min_Budget_Per_dic_q_task[min_Budget_Per_dic_q_sorted_id[j]]]
        # if min_Budget_Per_dic_q_sorted[i] > Budget_Task / sum_q:
        #     for z in range(i):
        #         client_selected_Undetermined.append(min_Budget_Per_dic_q_sorted_id[i])

        #找到k
        if min_Budget_Per_dic_q_sorted[i] > Budget_Task / sum_q:
            k_smallest = i
            Budget_Task_per_sum_q = Budget_Task / (sum_q - dic_q[min_Budget_Per_dic_q_sorted_id[k_smallest]][min_Budget_Per_dic_q_task[min_Budget_Per_dic_q_sorted_id[k_smallest]]])
            k_smallest_Budget_per_sum_q = Budget[min_Budget_Per_dic_q_sorted_id[k_smallest]][min_Budget_Per_dic_q_task[min_Budget_Per_dic_q_sorted_id[k_smallest]]] / dic_q[min_Budget_Per_dic_q_sorted_id[k_smallest]][min_Budget_Per_dic_q_task[min_Budget_Per_dic_q_sorted_id[k_smallest]]]
            # print("------k的值为：{}".format(k_smallest))
            # print("------平均单位质量的成本值为：{}".format(Budget_Task_per_sum_q))
            # print("------第k位客户端的平均单位质量的成本值为：{}".format(k_smallest_Budget_per_sum_q))
            break

    #选择前k个值
    for i in range(k_smallest):
        client_selected_Undetermined.append(min_Budget_Per_dic_q_sorted_id[i])
    # print("k的值为：{}".format(k_smallest))
    # print("选中{}个客户端".format(len(client_selected_Undetermined)))
    # print("选中但未分配的客户端为：{}".format(client_selected_Undetermined))

    min_keypayment = min(Budget_Task_per_sum_q, k_smallest_Budget_per_sum_q)
    #将各个客户端分给各个任务
    for i in range(len(client_selected_Undetermined)):

        if min_Budget_Per_dic_q_task[client_selected_Undetermined[i]] == 0:
            client_dict[0].append(client_selected_Undetermined[i])
            maxQTruthfulness_payment[0].append(dic_q[client_selected_Undetermined[i]][0] * min_keypayment)
            client_budget[0].append(Budget[client_selected_Undetermined[i]][0])

        if min_Budget_Per_dic_q_task[client_selected_Undetermined[i]] == 1:
            client_dict[1].append(client_selected_Undetermined[i])
            maxQTruthfulness_payment[1].append(dic_q[client_selected_Undetermined[i]][1] * min_keypayment)
            client_budget[1].append(Budget[client_selected_Undetermined[i]][1])

        if min_Budget_Per_dic_q_task[client_selected_Undetermined[i]] == 2:
            client_dict[2].append(client_selected_Undetermined[i])
            maxQTruthfulness_payment[2].append(dic_q[client_selected_Undetermined[i]][2] * min_keypayment)
            client_budget[2].append(Budget[client_selected_Undetermined[i]][2])

    # print("最终选中客户端为：{}".format(client_dict))
    # print("最终选中客户端报酬为：{}".format(maxQTruthfulness_payment))
    # print("最终选中客户端报价为：{}".format(client_budget))

    #用于检查
    # sum_payment = 0
    # for i in range(len(maxQTruthfulness_payment)):
    #     sum_payment += sum(maxQTruthfulness_payment[i])
    #     print("{}个任务的和为{}".format(i,sum(maxQTruthfulness_payment[i])))
    # print("共支付的价格为：{}".format(sum_payment))

    return client_dict, maxQTruthfulness_payment