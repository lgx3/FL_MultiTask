'''
数据划分：独立同分布

'''


import numpy as np

from utils import get_datasets

#数据独立同分布划分

'''
datasets： 提供常用的数据集加载，
设计上都是继承 torch.utils.data.Dataset，
主要包括 MNIST、CIFAR10/100、ImageNet、COCO等；

transforms：提供常用的数据预处理操作，
主要包括对 Tensor 以及 PIL Image 对象的操作；
'''

def mnist_fmnist_iid(dataset, num_users):
    """
    独立同分布
    对MNIST数据集采样数据(IID数据)
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #60000/20 = 3000
    num_items = int(len(dataset) / num_users)
    '''
        len(dataset)所有元素，= num_items * num_users
    '''
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]


    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,replace=False))
        '''
            dict_users[i] dict类型{'0':{1,3,4}}

            replace表示是否重用元素
            numpy.random.choice(a, size=None, replace=True, p=None)
            a : 如果是一维数组，就表示从这个一维数组中随机采样；如果是int型，就表示从0到a-1这个序列中随机采样
            从[0,1,2,3 ... len(dataset)]采样num_items个元素

            这很合理，dataset相当于矩阵，行为user，列为Item
            每个user为一行，列为item数量，所以对每个user采样num_item个元素

        '''
        all_idxs = list(set(all_idxs) - dict_users[i])
        '''
            set(all_idxs):{0,1,2,3,4,5,6,7...}
            每个user都减一次，最后为空
            函数返回dict_users：dict of image index
            dict_users[i]类似{'0':{1,3,4}}
        '''

    return dict_users


def cifar_iid(dataset, num_users):
    """
    和上面mnist_iid一样一样滴
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)#计算每个客户端可以有多少条数据
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]#dict_users:字典，用于存放客户端数据；all_idxs：列表：1~数据集总长度
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users#返回独立同分布的数据划分结果


#
# if __name__ == '__main__':
#
#     # 存放所有任务的数据集
#     test_dataset_list = []
#     train_dataset_list = []
#     Task = [0, 1]
#     dataset_name = ['fmnist', 'cifar10']
#     train_dataset, test_dataset = get_datasets('fmnist')
#     test_dataset_list.append(test_dataset)
#     train_dataset_list.append(train_dataset)
#     train_dataset, test_dataset = get_datasets('cifar10')
#     test_dataset_list.append(test_dataset)
#     train_dataset_list.append(train_dataset)
#
#     testDistribution = getTestDistribution(test_dataset_list)
#     print(testDistribution)
#
#     print(test_dataset_list)
#
#     #
    # mnist_train_dataset, mnist_test_dataset = get_datasets('fmnist')
    # mnist_dict_users = mnist_fmnist_iid(mnist_train_dataset,20)
    # # print("mnist独立同分布的数据划分结果为：{}".format(mnist_dict_users))
    # for i in range(len(mnist_dict_users)):
    #     print("mnist第{}个客户端总的数据量：{}".format(i, len(mnist_dict_users[i])))
    #
    # print("mnist测试集数据分布：{}".format(getTestDistribution(['fmnist'])))
    #
    #
    # cifar10_train_dataset, cifar10_test_dataset = get_datasets('cifar10')
    # cifar10_dict_users = cifar_iid(cifar10_train_dataset, 20)
    # # print("cifar10独立同分布的数据划分结果为：{}".format(cifar10_dict_users))
    # for i in range(len(cifar10_dict_users)):
    #     print("cifar10第{}个客户端总的数据量：{}".format(i, len(cifar10_dict_users[i])))
    # print("cifar10测试集数据分布：{}".format(getTestDistribution(['cifar10'])))
