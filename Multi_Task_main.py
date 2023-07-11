#主函数
import time
import copy
import pickle

import torch

from options import args_parser
from utils import get_datasets,average_weights
from FL_MultiTask.models import MLP,CNNMnist,CNNCifar,CNNFashion_Mnist

if __name__ == '__main__':
    #实验开始时间
    start_time = time.time()

    # # define paths
    # path_project = os.path.abspath('.')
    # logger = SummaryWriter('./logs')

    #取出参数
    args = args_parser()

    # 输出实验模型参数等细节
    #exp_details(args)

    '''
        if args.gpu_id:
            torch.cuda.set_device(args.gpu_id)
        device = 'cuda' if args.gpu else 'cpu'
        '''
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups  加载数据集和用户组?????????
    train_dataset, test_dataset = get_datasets(args)

    # BUILD MODEL  构建客户端训练模型
    if args.model == 'cnn':
        # Convolutional neural netork卷积神经网络
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        # torch.Size([3, 32, 32])
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)

    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.  设置模型进行训练，并将其发送给设备
    #global_model由上面的构建模型确定
    global_model.to(device)

    global_model.train()

    print(global_model)

    # copy weights 复制权值
    global_weights = global_model.state_dict()

    # Training  训练的损失和精确度
    train_loss, train_accuracy = [], []

    val_acc_list, net_list = [], []

    cv_loss, cv_acc = [], []

    print_every = 2

    val_loss_pre, counter = 0, 0

    # 客户端本地训练，epoch选为10（一个epoch指所有的数据送入网络完成一次前向计算和反向传播的过程）
    # args.epochs为options.py中的全局轮数，epoch[0,1,2,3,4,5,6,7,8,9]
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')  # 显示全局训练轮次
        global_model.train()
        '''
            如果模型中有BN层(Batch Normalization）和Dropout，
            需要在训练时添加model.train()。
            model.train()是保证BN层能够用到每一批数据的均值和方差。
            对于Dropout，model.train()是随机取一部分网络连接来训练更新参数
        '''

        # 随机选择10个用户  总用户*比例:frac=0.1  num_user=100   m=10
        m = max(int(args.frac * args.num_users), 1)  # 确定客户端数量m
        # idxs_users为拥有是个客户端的数组？
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 从0-99中选10个客户端，不能重复选择
        ''' 
            frac = 0.1
            num_users = 100
            dict_users[i] dict类型{'0':{1,3,4}}

            replace表示是否重用元素
            numpy.random.choice(a, size=None, replace=True, p=None)
            a : 如果是一维数组，就表示从这个一维数组中随机采样；如果是int型，
            就表示从0到a-1这个序列中随机采样
            从[0,1,2,3 ... len(dataset)]采样num_items个元素

            这很合理，dataset相当于矩阵，行为user，列为Item
            每个user为一行，列为item数量，所以对每个user采样num_item个元素

            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            这100个用户下标
            随机选择10个
        '''

        # 本地客户端训练，idx为下标，
        for idx in idxs_users:
            # 对于每个用户
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            '''
                idx的作用是什么？
                返回用户组：
                    dict类型{key:value}
                        key:用户的索引
                        value:这些用户的相应数据

                    user_group就是dict_users
                    dict_users[i]类似{'0':{1,3,4}}
            '''

            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            '''
                调用update.py中的update_weights函数
                for iter in range(self.args.local_ep)
                一个用户经过10个本地batch
                返回：
                return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
            '''

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        '''
            获得tensor的shape:
                test.shape
                print(local_weights[1]['layer_input.weight'].shape)
                torch.Size([64, 3072])
            3072=3*32*32

            (layer_input): Linear(in_features=3072, out_features=64, bias=True)
            fc1 = nn.Linear(16 * 5 * 5, 120) 权重却是torch.Size([120, 400])

            得出来的3072正好是上面的：
                args.model == 'mlp':

            local_weights[1]['layer_input.weight']:
                tensor([[ 0.0053, -0.0159,  0.0121,  ...,  0.0102,  0.0121,  0.0087],
                        [-0.0181,  0.0125,  0.0130,  ...,  0.0134,  0.0020,  0.0107],
                        ...,
                        [ 0.0023, -0.0054, -0.0015,  ...,  0.0022,  0.0147,  0.0071]])
        '''
        w_avg = copy.deepcopy(local_weights[0])
        # print(len(local_weights))  # 10

        '''
            更新全局权重
            传参为local_weights
            local_weights.append(copy.deepcopy(w))

            len(local_weights) = 10
        '''
        # print('传参local_weights')
        # print(len(local_weights))

        # 一个一个用户更新全局权重
        # 求全局模型的权重，求本地模型之和，再除以本地客户数量（平均）
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        '''
            torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
        '''

        # 求平均损失
        loss_avg = sum(local_losses) / len(local_losses)
        # 将平均损失加到训练损失集中
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        # 计算所有用户在每个epoch中的平均训练精度和损失
        list_acc, list_loss = [], []

        global_model.eval()

        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        '''
            每个用户推断的精度和损失存到list_acc、list_loss
            这里的list_loss没用
            用的是：
                train_loss.append(loss_avg)
        '''

        train_accuracy.append(sum(list_acc) / len(list_acc))

        '''每2轮输出全局训练损失 print_every=2
        if (epoch + 1) % print_every == 0:
        '''
        # 每轮输出每轮的全局训练损失
        if epoch >= 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')

            print(f'Training Loss : {np.mean(np.array(train_loss))}')

            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Test inference after completion of training
    # 完成所有轮训练的测试推断（推断损失）
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')

    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))

    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    plt.plot(epoch, train_accuracy)
    plt.show

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)





    print('\n Total Run Time:{0:0.4f}'.format(time.time()-start_time))#计算总的运行时间



