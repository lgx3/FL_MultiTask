#较为极端的数据训练的
#每个任务有自己单独的预算

import copy
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pyplot import MultipleLocator
from collections import defaultdict


from FL_MultiTask.models.Update import LocalUpdate
from FL_MultiTask.models.Nets import MLP, CNNMnist, CNNCifar10new
from FL_MultiTask.models.test import test_img
from util.client_selection import getClientBudget, getClientBudget_q, CLQM, bidPriceFirst_ClientSelection, \
    RandomClientSelect, Individual_Knapsack_greedy
from util.getClientQuality import getQ
from util.dirichlet import getTestDistribution, preferDistribution, exchangeDistribution
from options import args_parser
from utils import get_datasets


'''
联邦学习训练
输入：客户端列表：idx_users、数据划分样本结果dict_users、客户端样本数量权重p、训练集、测试集
'''

def fl_training(idxs_users, dict_users, p, dataset_train, dataset_test, args, task):
    print('进行{}数据集的训练'.format(task))
    print('参与训练的客户端是{}'.format(idxs_users))
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and task == 'cifar10':
        print("cnn训练")
        # net_glob = cnn1(num_classes=10).to(args.device)
        net_glob = CNNCifar10new(args=args).to(args.device)
        # net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and task == 'mnist' or task == 'fmnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        print("mlp训练")
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.train()
    # training
    loss_train = []
    acc_all = []

    if task == 'mnist':
        lr = 0.05
        local_bs = 10

    if task == 'fmnist':
        lr = 0.01
        local_bs = 10

    if task == 'cifar10':
        lr = 0.1
        local_bs = 100

    # lr_tp = args.lr
    lr_tp = lr
    for iter in range(args.epochs):
        loss_locals = []
        w_locals = []
        # 本都训练
        for idx in idxs_users:
            # print('客户端{}在训练'.format(idx))
            # local = LocalUpdate(args=args, class_weight=class_weight1[idx], dataset=dataset_train, idxs=dict_users[idx])
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], local_bs = local_bs)
            # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), lrr=lr)
            print('客户端{}在训练，loss为：{}'.format(idx,loss))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        w_glob = FedAvg(w_locals, p, idxs_users)
        # args.lr *= 0.93

        if task == 'cifar10':
            # lr *= 0.63
            lr *= 0.93
            # lr = lr * (0.333 ** iter)

        if task == 'fmnist':
            # lr *= 0.63
            # lr *= 0.998
            lr = lr * (0.998 ** iter)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test))
        acc_all.append(acc_test)

    # net_glob.eval()
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Testing accuracy: {:.2f}".format(acc_test))
    lr = lr_tp
    return acc_test, acc_all

def FedAvg(w, p, idxs_users):
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

'''
输入：某个任务中，客户端的数据分布
输出：每个客户端总的数据量占所有客户端数据量的比重，用于联邦学习中的聚合
'''
def getP(pic_distribution_every_client):
    P = {}
    index_client = 0
    #sum_data为所有客户端训练集数据量之和：比如：mnist = 60000
    sum_data = np.sum(pic_distribution_every_client).sum()
    # print("sum_data={}".format(sum_data))
    for distribution in pic_distribution_every_client:
        P[index_client] = np.sum(distribution) / sum_data
        index_client += 1
    return P


def main():
    # 获取参数
    args = args_parser()

    X_users = []
    for i in range(args.num_users):
        X_users.append(i + 1)

    # 画图工具,X代表迭代轮数
    X = []
    All_client = []
    for i in range(args.epochs):
        X.append(i + 1)

    for i in range(50):
        All_client.append(i)



    # 存放所有任务的数据集
    test_dataset_list = []
    train_dataset_list = []
    Task = [0, 1, 2]
    dataset_name = ['mnist','fmnist', 'cifar10']
    train_dataset_mnist, test_dataset_mnist = get_datasets('mnist')
    train_dataset_list.append(train_dataset_mnist)
    test_dataset_list.append(test_dataset_mnist)

    train_dataset_fmnist, test_dataset_fmnist = get_datasets('fmnist')
    train_dataset_list.append(train_dataset_fmnist)
    test_dataset_list.append(test_dataset_fmnist)

    train_dataset_cifar10, test_dataset_cifar10 = get_datasets('cifar10')
    train_dataset_list.append(train_dataset_cifar10)
    test_dataset_list.append(test_dataset_cifar10)

    print("待训练的任务为：{}".format(dataset_name))
    print(train_dataset_list)

    #任务预算
    Budget_Task = [30, 30, 35]
    # Budget_Task_all = 95

    #获取所有客户端的本地数据分布(所有任务的)：三个任务
    client_dict = []
    dict_users_mnist, distribution_mnist,class_weight_mnist = preferDistribution(train_dataset_mnist, args)
    dict_users_fmnist, distribution_fmnist,_ = preferDistribution(train_dataset_fmnist, args)
    dict_users_cifar10, distribution_cifar10,_ = preferDistribution(train_dataset_cifar10, args)
    client_dict.append(distribution_mnist)
    dict_users_fmnist, distribution_fmnist,class_weight_fmnist = exchangeDistribution(dict_users_fmnist, distribution_fmnist, 'fmnist')
    dict_users_cifar10, distribution_cifar10,class_weight_cifar10 = exchangeDistribution(dict_users_cifar10, distribution_cifar10, 'cifar10')
    client_dict.append(distribution_fmnist)
    client_dict.append(distribution_cifar10)
    print("每个客户端针对每个任务的数据量分布：{}".format(client_dict))
    print(type(client_dict))



















    # 获取测试集的数据分布
    testDistribution = getTestDistribution(test_dataset_list)
    # print("每个任务的测试集数据分布：{}".format(testDistribution))

    All_Client = defaultdict(list)
    for i in range(len(dataset_name)):
        All_Client[i] = All_client
    print(All_Client)

    '''——————————————————————EMD_MQFL：考虑数据量和emd时，客户端的选择情况——————————————————————'''
    # 考虑数据量和emd时，获取所有客户端的质量(所有任务的)
    dic_q = getQ(client_dict, testDistribution, args)
    print("客户端针对每个任务的质量为：", dic_q)  # 输出的为客户端所有任务的质量

    # 根据质量，获取客户端预算
    budget_q = defaultdict(list)
    budget_q_1 = getClientBudget_q(dic_q, args)
    budget_q_2 = getClientBudget(3, 50,args)
    for i in range(len(budget_q_1)):
        for j in range(len(budget_q_1[i])):
            budget_q[i].append(budget_q_1[i][j] + budget_q_2[i][j])

    print("客户端针对每个任务的报价为：", budget_q_2)

    print("\n*******************************************EMD-MQFL选择客户端*******************************************")
    client_selected_every_task_q, client_selected_every_task_payment_q = CLQM(dic_q, budget_q_2, Budget_Task, args)
    print("EMD-MQFL所有任务选中客户端集合：{}".format(client_selected_every_task_q))
    print("EMD-MQFL选中客户端支付价格：{}".format(client_selected_every_task_payment_q))

    sum_emd_payment = [0,0,0]
    for i in range(len(client_selected_every_task_payment_q)):
        for j in range(len(client_selected_every_task_payment_q[i])):
            sum_emd_payment[i] += client_selected_every_task_payment_q[i][j]

    print("sum_emd_payment = {}".format(sum_emd_payment))

    # 获取EMD-MQFL选中客户端的报价
    EMD_MQFL_selection_bid = defaultdict(list)
    for i in client_selected_every_task_q.keys():
        for j in range(len(client_selected_every_task_q[i])):
            EMD_MQFL_selection_bid[i].append(budget_q_2[client_selected_every_task_q[i][j]][i])
    print("EMD-MQFL选中客户端的报价：{}".format(EMD_MQFL_selection_bid))


    '''√'''
    print("\n*******************************************报价优先选择客户端*******************************************")
    Bidfirst_client_selected, Bidfirst_payment = bidPriceFirst_ClientSelection(budget_q_2, Budget_Task)
    print("报价优先选择时，所有任务选中的客户端集合：{}".format(Bidfirst_client_selected))
    print("报价优先选择时，所有任务选中的客户端支付价格：{}".format(Bidfirst_payment))

    '''√'''
    print("\n*******************************************背包贪婪选择客户端*******************************************")
    Knapsack_greedy_client_selected, Knapsack_greedy_payment = Individual_Knapsack_greedy(budget_q_2, client_dict, Budget_Task)
    print("背包贪婪选择时，所有任务选中的客户端集合：{}".format(Knapsack_greedy_client_selected))
    print("背包贪婪选择时，所有任务选中的客户端支付价格：{}".format(Knapsack_greedy_payment))

    '''√'''
    print("\n*******************************************随机选择客户端*******************************************")
    Random_client_selected, Random_payment = RandomClientSelect(budget_q_2, Budget_Task, args)
    print("随机选择时，所有任务选中的客户端集合：{}".format(Random_client_selected))
    print("随机选择时，所有任务选中的客户端支付价格：{}".format(Random_payment))


    #用于存放客户端数据质量，最后画图显示客户端数据质量
    Task_Quality_mnist_cnn = []
    Task_Quality_fmnist_cnn = []
    Task_Quality_cifar10_cnn = []

    for i in range(args.num_users):
        Task_Quality_mnist_cnn.append(dic_q[i][0])
        Task_Quality_fmnist_cnn.append(dic_q[i][1])
        Task_Quality_cifar10_cnn.append(dic_q[i][2])


    for task in Task:
        # 获取任务的名字，以数据集的名字命名
        task_name = dataset_name[task]
        dataset_train = train_dataset_list[task]
        dataset_test = test_dataset_list[task]

        # dic_users, _, _, _ = distribute_data_dirichlet(dataset_train, args)
        #传入任务对应的选中的客户端:client_dict = {0：[],1:[],2:[]}
        p = getP(client_dict[task])
        # print("{}的客户端数据分布为：{}".format(task,client_dict[task]))
        # print("{}的客户端的p为：{}".format(task,p))


        if task == 0:
            acc_test, acc_all_mnist = fl_training(client_selected_every_task_q[task], dict_users_mnist, p, dataset_train, dataset_test, args, dataset_name[task])
            print("EMD-MQFL选择时，mnist选中客户端参与训练的准确率：{}".format(acc_all_mnist))

            BidPrice_first_acc_test, BidPrice_first_acc_all_mnist = fl_training(Bidfirst_client_selected[task],dict_users_mnist, p, dataset_train,dataset_test, args, dataset_name[task])
            print("报价优先选择时，mnist选中客户端参与训练的准确率：{}".format(BidPrice_first_acc_all_mnist))

            Knapsack_greedy_acc_test, Knapsack_greedy_acc_all_mnist = fl_training(Knapsack_greedy_client_selected[task],dict_users_mnist, p, dataset_train,dataset_test, args, dataset_name[task])
            print("背包贪婪选择时，mnist选中客户端参与训练的准确率：{}".format(Knapsack_greedy_acc_all_mnist))

            Random_acc_test, Random_acc_all_mnist = fl_training(Random_client_selected[task],dict_users_mnist, p, dataset_train,dataset_test, args, dataset_name[task])
            print("随机选择时，mnist选中客户端参与训练的准确率：{}".format(Random_acc_all_mnist))


        if task == 1:
            acc_test, acc_all_fmnist = fl_training(client_selected_every_task_q[task], dict_users_fmnist, p, dataset_train, dataset_test, args, dataset_name[task])
            print("EMD-MQFL选择时，fmnist选中客户端参与训练的准确率：{}".format(acc_all_fmnist))

            BidPrice_first_acc_test, BidPrice_first_acc_all_fmnist = fl_training(Bidfirst_client_selected[task], dict_users_fmnist, p, dataset_train, dataset_test, args, dataset_name[task])
            print("报价优先选择时，fmnist选中客户端参与训练的准确率：{}".format(BidPrice_first_acc_all_fmnist))

            Knapsack_greedy_acc_test, Knapsack_greedy_acc_all_fmnist = fl_training(Knapsack_greedy_client_selected[task],dict_users_fmnist, p, dataset_train,dataset_test, args, dataset_name[task])
            print("背包贪婪选择时，fmnist选中客户端参与训练的准确率：{}".format(Knapsack_greedy_acc_all_fmnist))

            Random_acc_test, Random_acc_all_fmnist = fl_training(Random_client_selected[task],dict_users_fmnist, p, dataset_train,dataset_test, args, dataset_name[task])
            print("随机选择时，fmnist选中客户端参与训练的准确率：{}".format(Random_acc_all_fmnist))


        if task == 2:
            acc_test, acc_all_cifar10 = fl_training(client_selected_every_task_q[task], dict_users_cifar10, p, dataset_train,dataset_test, args, dataset_name[task])
            print("EMD-MQFL选择时，cifar10选中客户端参与训练的准确率：{}".format(acc_all_cifar10))

            BidPrice_first_acc_test, BidPrice_first_acc_all_cifar10 = fl_training(Bidfirst_client_selected[task],dict_users_cifar10, p, dataset_train,dataset_test, args, dataset_name[task])
            print("报价优先选择时，cifar10选中客户端参与训练的准确率：{}".format(BidPrice_first_acc_all_cifar10))

            Knapsack_greedy_acc_test, Knapsack_greedy_acc_all_cifar10 = fl_training(Knapsack_greedy_client_selected[task],dict_users_cifar10, p, dataset_train,dataset_test, args,dataset_name[task])
            print("背包贪婪选择时，cifar10选中客户端参与训练的准确率：{}".format(Knapsack_greedy_acc_all_cifar10))

            Random_acc_test, Random_acc_all_cifar10 = fl_training(Random_client_selected[task],dict_users_cifar10, p, dataset_train,dataset_test, args,dataset_name[task])
            print("随机选择时，cifar10选中客户端参与训练的准确率：{}".format(Random_acc_all_cifar10))




    '''——————————————————————求精度平均值，用于画图展示——————————————————————'''
    Acc_avg_EMD_MQFL = []#acc_all_fmnist、acc_all_cifar10
    Acc_avg_BidPrice_first = []#BidPrice_first_acc_all_fmnist、BidPrice_first_acc_all_fmnist
    Acc_avg_Knapsack_greedy = []
    Acc_avg_Random = []



    for i in range(len(X)):
        Acc_avg_EMD_MQFL.append((acc_all_mnist[i] + acc_all_fmnist[i] + acc_all_cifar10[i]) / 3)
        Acc_avg_BidPrice_first.append((BidPrice_first_acc_all_mnist[i] + BidPrice_first_acc_all_fmnist[i] + BidPrice_first_acc_all_cifar10[i]) / 3)
        Acc_avg_Knapsack_greedy.append((Knapsack_greedy_acc_all_mnist[i] + Knapsack_greedy_acc_all_fmnist[i] + Knapsack_greedy_acc_all_cifar10[i]) / 3)
        Acc_avg_Random.append((Random_acc_all_mnist[i] + Random_acc_all_fmnist[i]+Random_acc_all_cifar10[i])/ 3)


    print("EMD_MQFL每个任务的平均精度：{}".format(Acc_avg_EMD_MQFL))
    print("BidPrice_first每个任务的平均精度：{}".format(Acc_avg_BidPrice_first))
    print("Knapsack_greedy每个任务的平均精度：{}".format(Acc_avg_Knapsack_greedy))
    print("Random每个任务的平均精度：{}".format(Acc_avg_Random))


    '''——————————————————————画质量图——————————————————————'''
    plt.plot(X_users, Task_Quality_mnist_cnn, 'b:', marker='s', ms=3, label='mnist_cnn')
    plt.plot(X_users, Task_Quality_fmnist_cnn, 'r:', marker='o', ms=3,label='fmnist_cnn')
    plt.plot(X_users, Task_Quality_cifar10_cnn, 'g:', marker='*', ms=3,label='cifar10_cnn')
    plt.xlabel('Clients')
    plt.ylabel('Data_Quality')
    plt.xlim(0, 50)
    # plt.ylim(0,1)
    plt.xticks(range(0, 50, 1))
    # plt.yticks(range(0,1,0.1))
    plt.grid()
    plt.legend()
    plt.savefig('D:/Code/WorkPlace/pythonWork/FL_MultiTask/save/Individual_Budget_FederatedLearning/20230313_实验1_0.2_5_TaskQuality.png')
    plt.show()

    '''——————————————————————Mnist画图——————————————————————'''
    plt.plot(X, acc_all_mnist, 'r:', marker='o', ms=3,label='EMD_MQFL')
    plt.plot(X, BidPrice_first_acc_all_mnist, 'g:', marker='*', ms=3,label='Bid_Price_First')
    plt.plot(X, Knapsack_greedy_acc_all_mnist, 'k:', marker='P', ms=3,label='Knapsack_greedy')
    plt.plot(X, Random_acc_all_mnist, 'b:', marker='s', ms=3,label='Random')
    plt.title('Mnist')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")

    # 设置坐标间隔
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(2.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0)

    plt.grid()











    plt.legend()
    plt.savefig('D:/Code/WorkPlace/pythonWork/FL_MultiTask/save/Individual_Budget_FederatedLearning/20230313_实验1_0.2_5_mnist.png')
    plt.show()

    '''——————————————————————Fmnist画图——————————————————————'''
    plt.plot(X, acc_all_fmnist, 'r:', marker='o', ms=3,label='EMD_MQFL')
    plt.plot(X, BidPrice_first_acc_all_fmnist, 'g:', marker='*', ms=3,label='Bid_Price_First')
    plt.plot(X, Knapsack_greedy_acc_all_fmnist, 'k:', marker='P', ms=3,label='Knapsack_greedy')
    plt.plot(X, Random_acc_all_fmnist, 'b:', marker='s', ms=3,label='Random')
    plt.title('Fmnist')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")

    # 设置坐标间隔
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0)

    plt.grid()
    plt.legend()
    plt.savefig('D:/Code/WorkPlace/pythonWork/FL_MultiTask/save/Individual_Budget_FederatedLearning/20230313_实验1_0.2_5_fmnist.png')
    plt.show()

    '''——————————————————————Cifar10画图——————————————————————'''
    plt.plot(X, acc_all_cifar10, 'r:', marker='o', ms=3,label='EMD_MQFL')
    plt.plot(X, BidPrice_first_acc_all_cifar10, 'g:', marker='*', ms=3,label='Bid_Price_First')
    plt.plot(X, Knapsack_greedy_acc_all_cifar10, 'k:', marker='P', ms=3,label='Knapsack_greedy')
    plt.plot(X, Random_acc_all_cifar10, 'b:', marker='s', ms=3,label='Random')
    plt.title('Cifar10')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    #plt.suptitle('Experiments')

    #设置坐标间隔
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0)

    plt.grid()
    plt.legend()
    plt.savefig('D:/Code/WorkPlace/pythonWork/FL_MultiTask/save/Individual_Budget_FederatedLearning/20230313_实验1_0.2_5_cifar10.png')
    plt.show()


    '''——————————————————————按照迭代次数，精度平均值画图——————————————————————'''
    #每次任务的精度

    plt.plot(X, Acc_avg_EMD_MQFL, 'r:', marker='o', ms=3,label='EMD_MQFL')
    plt.plot(X, Acc_avg_BidPrice_first, 'g:', marker='*', ms=3,label='Bid_Price_First')
    plt.plot(X, Acc_avg_Knapsack_greedy, 'k:', marker='P', ms=3,label='Knapsack_greedy')
    plt.plot(X, Acc_avg_Random, 'b:', marker='s', ms=3,label='Random')

    plt.title('Average_Accuracy')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")

    # 设置坐标间隔
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(2.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0)


    plt.grid()
    plt.legend()
    plt.savefig('D:/Code/WorkPlace/pythonWork/FL_MultiTask/save/Individual_Budget_FederatedLearning/20230313_实验1_0.2_5_算法平均精度.png')
    plt.show()




if __name__ == '__main__':
    main()
