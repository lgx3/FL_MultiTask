import copy
import numpy as np

from FL_MultiTask.models.Update import LocalUpdate
from FL_MultiTask.models.Nets import MLP, CNNMnist, cnn1
from FL_MultiTask.models.test import test_img
from util.dirichlet import getAllClientDataDistribution, preferDistribution, exchangeDistribution
from options import args_parser
from utils import get_datasets


'''
联邦学习训练
'''

def fl_training(idxs_users, dict_users, p, dataset_train, dataset_test, args, task,class_weight1):
    print('进行{}数据集的训练'.format(task))
    print('参与训练的客户端是{}'.format(idxs_users))
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and task == 'cifar10':
        net_glob = cnn1(num_classes=10).to(args.device)
    elif args.model == 'cnn' and task == 'mnist' or task == 'fmnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
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
    for iter in range(args.epochs):
        loss_locals = []
        w_locals = []
        # 本都训练
        for idx in idxs_users:
            # print('客户端{}在训练'.format(idx))
            local = LocalUpdate(args=args, class_weight=class_weight1[idx], dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            print('客户端{}在训练，loss为：{}'.format(idx,loss))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        w_glob = FedAvg(w_locals, p, idxs_users)
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


def getP(pic_distribution_every_client):
    P = {}
    index_client = 0
    sum_data = np.sum(pic_distribution_every_client).sum()
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

    #获取所有客户端的本地数据分布(所有任务的)：三个任务
    client_dict = []
    dict_users_mnist_1, distribution_mnist_1,class_weight_mnist = preferDistribution(train_dataset_mnist, args)
    dict_users_fmnist_1, distribution_fmnist_1,_ = preferDistribution(train_dataset_fmnist, args)
    dict_users_cifar10_1, distribution_cifar10_1,_ = preferDistribution(train_dataset_cifar10, args)
    # client_dict.append(distribution_mnist)


    dict_users_mnist_2,client_dict_mnist_2 = getAllClientDataDistribution(['mnist'], args, n_class=10)
    dict_users_fmnist_2,client_dict_fmnist_2 = getAllClientDataDistribution(['fmnist'], args, n_class=10)
    dict_users_cifar10_2,client_dict_cifar10_2 = getAllClientDataDistribution(['cifar10'], args, n_class=10)


    dict_users_mnist = []
    dict_users_fmnist = []
    dict_users_cifar10 = []



    for i in range(len(distribution_mnist_1)):
        for j in range(len(distribution_mnist_1[i])):
            dict_users_mnist


    dict_users_fmnist, distribution_fmnist, class_weight_fmnist = exchangeDistribution(dict_users_fmnist,distribution_fmnist,'fmnist')
    dict_users_cifar10, distribution_cifar10, class_weight_cifar10 = exchangeDistribution(dict_users_cifar10,distribution_cifar10,'cifar10')


    # client_dict.append(distribution_fmnist)
    # client_dict.append(distribution_cifar10)
    # print("每个客户端针对每个任务的数据量分布：{}".format(client_dict))



    # # 获取测试集的数据分布
    # testDistribution = getTestDistribution(test_dataset_list)
    # # print("每个任务的测试集数据分布：{}".format(testDistribution))
    #
    # All_Client = defaultdict(list)
    # for i in range(len(dataset_name)):
    #     All_Client[i] = All_client
    # print(All_Client)
    #
    # '''——————————————————————EMD_MQFL：考虑数据量和emd时，客户端的选择情况——————————————————————'''
    # # 考虑数据量和emd时，获取所有客户端的质量(所有任务的)
    # dic_q = getQ(client_dict, testDistribution, args)
    # print("客户端针对每个任务的质量为：", dic_q)  # 输出的为客户端所有任务的质量
    #
    # # 根据质量，获取客户端预算
    # budget_q = defaultdict(list)
    # budget_q_1 = getClientBudget_q(dic_q, args)
    # budget_q_2 = getClientBudget(3, 50,args)
    # for i in range(len(budget_q_1)):
    #     for j in range(len(budget_q_1[i])):
    #         budget_q[i].append(budget_q_1[i][j] + budget_q_2[i][j])
    #
    # print("客户端针对每个任务的报价为：", budget_q)
    #
    # print("\n*******************************************EMD-MQFL选择客户端执行开始*******************************************\n")
    # client_selected_every_task_q, client_selected_every_task_payment_q = CLQM(dic_q, budget_q, Budget_Task,args)
    # print("\n*******************************************EMD-MQFL选择客户端执行结束*******************************************\n")
    #
    #
    #
    # '''——————————————————————报价优先选择的客户端——————————————————————'''
    # BidPrice_first_Selected, BidPrice_first_Selected_Payment = bidPriceFirst_ClientSelection(budget_q,Budget_Task)
    #
    #
    # '''——————————————————————随机选择的客户端——————————————————————'''
    # Random_Selected,Random_Selected_Payment = RandomClientSelect(budget_q,Budget_Task,args)
    #
    #
    #
    # print("\n************************最终选择结果************************")
    # print("EMD-MQFL所有任务选中客户端集合：{}".format(client_selected_every_task_q))
    # # print("EMD-MQFL选中客户端支付价格：{}".format(client_selected_every_task_payment_q))
    #
    # client_selected_every_task_payment_q_sort = defaultdict(list)
    # Key = list(client_selected_every_task_payment_q.keys())
    # for i in range(len(client_selected_every_task_payment_q)):
    #     for j in range(len(client_selected_every_task_payment_q)):
    #         if i == Key[j]:
    #             for z in range(len(client_selected_every_task_payment_q[i])):
    #                 client_selected_every_task_payment_q_sort[i].append(client_selected_every_task_payment_q[i][z])
    # print("EMD-MQFL选中客户端支付价格：{}".format(client_selected_every_task_payment_q_sort))
    #
    # #获取EMD-MQFL选中客户端的报价
    # EMD_MQFL_selection_bid = defaultdict(list)
    # for i in range(len(client_selected_every_task_q)):
    #     for j in range(len(client_selected_every_task_q[i])):
    #         EMD_MQFL_selection_bid[i].append(budget_q[client_selected_every_task_q[i][j]][i])
    # print("EMD-MQFL选中客户端的报价：{}".format(EMD_MQFL_selection_bid))
    #
    #
    # print("\n报价优先选择时，所有任务选中的客户端集合：{}".format(BidPrice_first_Selected))
    # print("报价优先选择时，所有任务选中的客户端支付价格：{}".format(BidPrice_first_Selected_Payment))
    #
    # print("\n随机选择时，所有任务选中的客户端集合：{}".format(Random_Selected))
    # print("随机选择时，所有任务选中的客户端支付价格：{}".format(Random_Selected_Payment))
    #
    #
    # #用于存放客户端数据质量，最后画图显示客户端数据质量
    # Task_Quality_mnist_cnn = []
    # Task_Quality_fmnist_cnn = []
    # Task_Quality_cifar10_cnn = []
    #
    # for i in range(args.num_users):
    #     Task_Quality_mnist_cnn.append(dic_q[i][0])
    #     Task_Quality_fmnist_cnn.append(dic_q[i][1])
    #     Task_Quality_cifar10_cnn.append(dic_q[i][2])
    #
    #
    #
    # for task in Task:
    #     # 获取任务的名字，以数据集的名字命名
    #     task_name = dataset_name[task]
    #     dataset_train = train_dataset_list[task]
    #     dataset_test = test_dataset_list[task]
    #
    #     dic_users, _, _, _ = distribute_data_dirichlet(dataset_train, args)
    #     p = getP(client_dict[task])
    #
    #
    #     if task == 0:
    #         acc_test, acc_all_mnist = fl_training(client_selected_every_task_q[task], dic_users, p, dataset_train, dataset_test, args, dataset_name[task],class_weight_mnist)
    #         print("考虑数据量和emd时，mnist选中客户端参与训练的准确率：{}".format(acc_all_mnist))
    #
    #         BidPrice_first_acc_test, BidPrice_first_acc_all_mnist = fl_training(BidPrice_first_Selected[task], dict_users_mnist, p, dataset_train, dataset_test, args, dataset_name[task],class_weight_mnist)
    #         print("报价优先时，mnist选中客户端参与训练的准确率：{}".format(BidPrice_first_acc_all_mnist))
    #
    #         Random_acc_test, Random_acc_all_mnist = fl_training(Random_Selected[task], dict_users_mnist, p, dataset_train, dataset_test, args, dataset_name[task],class_weight_mnist)
    #         print("随机选择时，mnist选中客户端参与训练的准确率：{}".format(Random_acc_all_mnist))
    #
    #     #     # all_acc_test_fmnist,all_acc_all_fmnist = fl_training(All_Client[task], dic_users, p, dataset_train, dataset_test, args, dataset_name[task])
    #     #     # print("所有客户端都参与时，，fmnist选中客户端参与训练的准确率：{}".format(all_acc_all_fmnist))
    #     #
    #     if task == 1:
    #         acc_test, acc_all_fmnist = fl_training(client_selected_every_task_q[task], dict_users_fmnist, p, dataset_train, dataset_test, args, dataset_name[task],class_weight_fmnist)
    #         print("考虑数据量和emd时，fmnist选中客户端参与训练的准确率：{}".format(acc_all_fmnist))
    #
    #         BidPrice_first_acc_test, BidPrice_first_acc_all_fmnist = fl_training(BidPrice_first_Selected[task], dict_users_fmnist, p, dataset_train, dataset_test, args, dataset_name[task],class_weight_fmnist)
    #         print("报价优先时，fmnist选中客户端参与训练的准确率：{}".format(BidPrice_first_acc_all_fmnist))
    #
    #         Random_acc_test, Random_acc_all_fmnist = fl_training(Random_Selected[task], dict_users_fmnist, p, dataset_train, dataset_test, args, dataset_name[task],class_weight_fmnist)
    #         print("随机选择时，fmnist选中客户端参与训练的准确率：{}".format(Random_acc_all_fmnist))
    #
    #     #     # all_acc_test_fmnist,all_acc_all_fmnist = fl_training(All_Client[task], dic_users, p, dataset_train, dataset_test, args, dataset_name[task])
    #     #     # print("所有客户端都参与时，，fmnist选中客户端参与训练的准确率：{}".format(all_acc_all_fmnist))
    #
    #     if task == 2:
    #         acc_test, acc_all_cifar10 = fl_training(client_selected_every_task_q[task], dict_users_cifar10, p, dataset_train,dataset_test, args, dataset_name[task],class_weight_cifar10)
    #         print("考虑数据量和emd时，cifar10选中客户端参与训练的准确率：{}".format(acc_all_cifar10))
    #
    #         BidPrice_first_acc_test, BidPrice_first_acc_all_cifar10 = fl_training(BidPrice_first_Selected[task],dict_users_cifar10, p, dataset_train,dataset_test, args, dataset_name[task],class_weight_cifar10)
    #         print("报价优先时，cifar10选中客户端参与训练的准确率：{}".format(BidPrice_first_acc_all_cifar10))
    #
    #         Random_acc_test, Random_acc_all_cifar10 = fl_training(Random_Selected[task],dict_users_cifar10, p, dataset_train,dataset_test, args, dataset_name[task],class_weight_cifar10)
    #         print("随机选择时，cifar10选中客户端参与训练的准确率：{}".format(Random_acc_all_cifar10))
    #
    #         # all_acc_test_cifar10, all_acc_all_cifar10 = fl_training(All_Client[task], dict_users_cifar10, p, dataset_train,dataset_test, args, dataset_name[task],class_weight_cifar10)
    #         # print("所有客户端都参与时，cifar10选中客户端参与训练的准确率：{}".format(all_acc_all_cifar10))
    #
    #
    #
    #
    # '''——————————————————————求精度平均值，用于画图展示——————————————————————'''
    #
    # Acc_avg_EMD_MQFL = []#acc_all_fmnist、acc_all_cifar10
    # Acc_avg_BidPrice_first = []#BidPrice_first_acc_all_fmnist、BidPrice_first_acc_all_fmnist
    # Acc_avg_Random = []
    # for i in range(len(X)):
    #     Acc_avg_EMD_MQFL.append((acc_all_mnist[i] + acc_all_fmnist[i] + acc_all_cifar10[i])/3)
    #     Acc_avg_BidPrice_first.append((BidPrice_first_acc_all_mnist[i] + BidPrice_first_acc_all_fmnist[i] + BidPrice_first_acc_all_cifar10[i]) / 3)
    #     Acc_avg_Random.append((Random_acc_all_mnist[i] + Random_acc_all_fmnist[i]+Random_acc_all_cifar10[i])/ 3)
    #
    #
    # print("EMD_MQFL每个任务的平均精度：{}".format(Acc_avg_EMD_MQFL))
    # print("BidPrice_first每个任务的平均精度：{}".format(Acc_avg_BidPrice_first))
    # print("Random每个任务的平均精度：{}".format(Acc_avg_Random))
    #
    #
    # '''——————————————————————画质量图——————————————————————'''
    # plt.plot(X_users, Task_Quality_mnist_cnn, 'b:', marker='s', ms=5, label='mnist_cnn')
    # plt.plot(X_users, Task_Quality_fmnist_cnn, 'r:', marker='o', ms=5, label='fmnist_cnn')
    # plt.plot(X_users, Task_Quality_cifar10_cnn, 'g:', marker='*', ms=5, label='cifar10_cnn')
    # plt.xlabel('Clients')
    # plt.ylabel('Data_Quality')
    # plt.xlim(0, 50)
    # # plt.ylim(0,1)
    # plt.xticks(range(0, 50, 1))
    # # plt.yticks(range(0,1,0.1))
    # plt.grid()
    # plt.legend()
    # plt.savefig('D:/Federated_Learning_experiments/FL_MultiTask/save/Task3_experiment_conclusion/20230224_实验1_0.5_2_TaskQuality.png',dpi=600)
    # plt.show()
    #
    # '''——————————————————————Mnist画图——————————————————————'''
    # plt.plot(X, acc_all_mnist, 'r:', marker='o', label='EMD_MQFL')
    # plt.plot(X, BidPrice_first_acc_all_mnist, 'g:', marker='*', label='Bid_Price_First')
    # plt.plot(X, Random_acc_all_mnist, 'k:', marker='P', label='Random')
    # plt.title('Mnist')
    # plt.xlabel("epochs")
    # plt.ylabel("Accuracy")
    #
    # # 设置坐标间隔
    # x_major_locator = MultipleLocator(1)
    # y_major_locator = MultipleLocator(5)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    #
    # plt.grid()
    # plt.legend()
    # plt.savefig('D:/Federated_Learning_experiments/FL_MultiTask/save/Task3_experiment_conclusion/20230224_实验1_0.5_2_mnist.png',dpi=400)
    # plt.show()
    #
    # '''——————————————————————Fmnist画图——————————————————————'''
    # plt.plot(X, acc_all_fmnist,'r:',marker='o', label='EMD_MQFL')
    # # plt.plot(X, without_d_acc_all_fmnist,'b:',marker='s', label='EMD_MQFL_without_d')
    # plt.plot(X, BidPrice_first_acc_all_fmnist,'g:',marker='*', label='Bid_Price_First')
    # plt.plot(X, Random_acc_all_fmnist,'k:',marker='P', label='Random')
    # plt.title('Fmnist')
    # plt.xlabel("epochs")
    # plt.ylabel("Accuracy")
    #
    # # 设置坐标间隔
    # x_major_locator = MultipleLocator(1)
    # y_major_locator = MultipleLocator(5)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    #
    # plt.grid()
    # plt.legend()
    # plt.savefig('D:/Federated_Learning_experiments/FL_MultiTask/save/Task3_experiment_conclusion/20230224_实验1_0.5_2_fmnist.png', dpi=400)
    # plt.show()
    #
    # '''——————————————————————Cifar10画图——————————————————————'''
    # plt.plot(X, acc_all_cifar10,'r:',marker='o', label='EMD-MQFL')
    # # plt.plot(X, without_d_acc_all_cifar10, 'b:', marker='s', label='EMD_MQFL_without_d')
    # plt.plot(X, BidPrice_first_acc_all_cifar10,'g:',marker='*', label='Bid_Price_First')
    # plt.plot(X, Random_acc_all_cifar10,'k:',marker='P', label='Random')
    # plt.title('Cifar10')
    # plt.xlabel("epochs")
    # plt.ylabel("Accuracy")
    # #plt.suptitle('Experiments')
    #
    # #设置坐标间隔
    # x_major_locator = MultipleLocator(1)
    # y_major_locator = MultipleLocator(5)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    #
    #
    # plt.grid()
    # plt.legend()
    # plt.savefig('D:/Federated_Learning_experiments/FL_MultiTask/save/Task3_experiment_conclusion/20230224_实验1_0.5_2_cifar10.png', dpi=400)
    # plt.show()
    #
    #
    # '''——————————————————————按照迭代次数，精度平均值画图——————————————————————'''
    # #每次任务的精度
    #
    # plt.plot(X, Acc_avg_EMD_MQFL, 'r:', marker='o', label='EMD_MQFL')
    # # plt.plot(X, without_d_acc_all_fmnist,'b:',marker='s', label='EMD_MQFL_without_d')
    # plt.plot(X, Acc_avg_BidPrice_first, 'g:', marker='*', label='Bid_Price_First')
    # plt.plot(X, Acc_avg_Random, 'k:', marker='P', label='Random')
    # plt.title('Average_Accuracy')
    # plt.xlabel("epochs")
    # plt.ylabel("Accuracy")
    #
    # # 设置坐标间隔
    # x_major_locator = MultipleLocator(1)
    # y_major_locator = MultipleLocator(5)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    #
    # plt.grid()
    # plt.legend()
    # plt.savefig('D:/Federated_Learning_experiments/FL_MultiTask/save/Task3_experiment_conclusion/20230224_实验1_0.5_2_算法平均精度.png',dpi=400)
    # plt.show()
    #
    #


if __name__ == '__main__':
    main()
