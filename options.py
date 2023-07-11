# #保存各个参数

import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()

    #联邦学习相关参数
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")#全局训练轮数
    parser.add_argument('--num_users', type=int,  default=50, help="number of users: K")#联邦学习用户端个数
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")#本地训练轮数
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")#本地batch size
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # 模型参数
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    #数据划分参数
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")#数据集名
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')#数据集类别数
    parser.add_argument('--num_channels', type=int, default=1, help='number of channels of imges')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--device', type=str, default='cuda:0', help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=8, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--concent', type=float, default=0.8,
                        help="concentration of dirichlet dist when doing niid sampling")

    args = parser.parse_args()
    # args.device = torch.device('dml' if args.gpu != -1 else 'cpu')

    # args = parser.parse_args([])
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    return args