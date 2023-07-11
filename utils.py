#获取数据集,返回训练集、测试集

import torch
from torchvision import datasets,transforms

def get_datasets(data,augment=True):
    train_dataset, test_dataset = None, None
    data_dir = './data/cifar'

    if data == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        train_dataset = datasets.MNIST('D:/Federated_Learning_experiments/FL_MultiTask/data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('D:/Federated_Learning_experiments/FL_MultiTask/data', train=False, download=True, transform=transform)

    elif data == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        train_dataset = datasets.FashionMNIST('D:/Federated_Learning_experiments/FL_MultiTask/data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('D:/Federated_Learning_experiments/FL_MultiTask/data', train=False, download=True, transform=transform)

    elif data == 'cifar10':
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10('D:/Federated_Learning_experiments/FL_MultiTask/data/cifar', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('D:/Federated_Learning_experiments/FL_MultiTask/data/cifar', train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)

    return train_dataset,test_dataset

def average_weights(w):
    """
    Returns the average of the weights.
    w应该是数组📕🥧
    """
    w_avg = copy.deepcopy(w[0])
    '''
        deepcopy函数：
        test = torch.randn(4, 4)
        print(test)
            tensor([[ 1.8693, -0.3236, -0.3465,  0.9226],
            [ 0.0369, -0.5064,  1.1233, -0.7962],
            [-0.5229,  1.0592,  0.4072, -1.2498],
            [ 0.2530, -0.4465, -0.8152, -0.9206]])
        w = copy.deepcopy(test[0])
        print(w)
            tensor([ 1.8693, -0.3236, -0.3465,  0.9226])
    '''
    # print('++++++++')
    # print(w)
    # print('=====')
    # print(w_avg)
    # print('++++++++++++++++++')
    # print(len(w)) == 10
    # 这个函数接受的是list类型的local_weights
    for key in w_avg.keys():
        for i in range(1, len(w)):
            # range(1, 10):1,2,3,4,5,6,7,8,9
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
        '''
            所有元素之和除以w的大小
            w是什么类型来着？？？
        '''
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    '''
        epoch:一个epoch指代所有的数据送入网络中完成一次前向
        计算及反向传播的过程，由于一个epoch常常太大，我们会
        将它分成几个较小的batches。
    '''
    return