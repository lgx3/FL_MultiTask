from util.dirichlet import preferDistribution, exchangeDistribution
from options import args_parser
from utils import get_datasets
from FL_MultiTask.models.Update import LocalUpdate
from FL_MultiTask.models.Nets import MLP, CNNMnist, cnn1
from FL_MultiTask.models.test import test_img
import copy
import numpy as np

def fl_training(idxs_users, dict_users, p, dataset_train, dataset_test, args, task):
    print('进行{}数据集的训练'.format(task))
    print('参与训练的客户端是{}'.format(idxs_users))
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and task == 'cifar10':
        net_glob = cnn1(num_classes=10).to(args.device)
        # net_glob = CNNCifar(args=args).to(args.device)
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
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
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
    #sum_data为所有客户端训练集数据量之和：比如：mnist = 60000
    sum_data = np.sum(pic_distribution_every_client).sum()
    # print("sum_data={}".format(sum_data))
    for distribution in pic_distribution_every_client:
        P[index_client] = np.sum(distribution) / sum_data
        index_client += 1
    return P
args = args_parser()
train_dataset_fmnist, test_dataset_fmnist = get_datasets('fmnist')
dict_users_fmnist, distribution_fmnist,_ = preferDistribution(train_dataset_fmnist, args)
dict_users_fmnist, distribution_fmnist, class_weight_fmnist = exchangeDistribution(dict_users_fmnist,distribution_fmnist, 'fmnist')
p = getP(distribution_fmnist)
users = [16, 17, 18, 19, 20, 21, 4, 2, 44, 36, 47, 1]
# users = [21, 32, 34, 12, 26]
for u in users:
    print('{}:{}'.format(u, distribution_fmnist[u]))
for u in users:
    print('{}:{}'.format(u, p[u]))
acc_test, acc_all = fl_training(users, dict_users_fmnist, p, train_dataset_fmnist, test_dataset_fmnist, args, 'fmnist')
print(acc_all)




