
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """


    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # sampling中idxs = np.arange(60000)
        # [1, 2]-->>tensor([1, 2])
        return torch.tensor(image), torch.tensor(label)

#本地更新
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        # 本地更新
        self.args = args
        self.logger = logger  # 记录器
        self.trainloader, self.validloader, self.testloader = \
            self.train_val_test(dataset, list(idxs))

        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        '''
            默认标准设置为NLL损失函数
            类似：net = net.to(device)
            CrossEntropyLoss()=log_softmax() + NLLLoss() 
            softmax(x)+log(x)+nn.NLLLoss====>nn.CrossEntropyLoss
            其中softmax函数又称为归一化指数函数，它可以把一个多维向量压缩在（0，1）之间，并且它们的和为1
        '''

    def train_val_test(self, dataset, idxs):
        """
        上面调用了这个函数
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.

        给定一个数据集返回：训练集，验证集，测试集
        对于一个给定数据集和用户索引
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]
        '''
            划分索引比例(80,10,10)


        '''

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        '''
            shuffle设置为True每一个epoch都重新洗牌数据
            DataLoader(training_data, batch_size=64, shuffle=True)    
        '''
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val) / 10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        '''
            更新权重
        '''
        model.train()
        '''
            告诉我们的网络，这个阶段是用来训练的，可以更新参数
        '''
        epoch_loss = []

        # Set optimizer for the local updates
        '''
            为本地更新选择优化器
        '''
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # epoch
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                '''
                    batch_idx
                    (images, labels)
                    enumerate(self.trainloader)会将训练集数据一个
                    batch一个batch地取出来用来训练
                    使用enumerate进行dataloader中的数据读取用于
                    神经网络的训练是第一种数据读取方法，其基本形式
                    即为for index， item in enumerate(dataloader['train'])，
                    其中item中0为数据，1为label.
                '''
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                '''
                    将模型的参数梯度设为0
                '''
                log_probs = model(images)
                '''
                    获得前项传播结果
                    model函数在调用的时候会调用call，于是调用forward。
                '''
                loss = self.criterion(log_probs, labels)
                '''
                    这个代码就是交叉熵：log_softmax()+NLLLoss()

                    from torch import nn

                    self.criterion = nn.NLLLoss().to(self.device)
                    loss=nn.NLLLoss()
                    loss(input,target)

                    CrossEntropyLoss()=log_softmax() + NLLLoss() 
                    softmax(x)+log(x)+nn.NLLLoss====>nn.CrossEntropyLoss

                '''
                loss.backward()
                '''
                    深度学习中最为重要的反向传播计算，
                    pytorch用非常简单的backward()函数就实现了
                '''
                optimizer.step()
                '''
                    所有的optimizer都实现了step()方法，
                    这个方法会更新所有的参数
                '''

                if self.args.verbose and (batch_idx % 10 == 0):
                    '''
                        for batch_idx, (images, labels) in enumerate(self.trainloader):
                    '''
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round,
                        iter,
                        batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item()))
                    '''
                        batch_idx为batch数量吗？
                        len(images)为一个batch中的数据即batch_size
                        batch_idx * len(images)为数据量



                    '''
                    '''
                        全局轮数：
                            本地epoch:
                                损失loss：
                    '''
                self.logger.add_scalar('loss', loss.item())
                '''
                    add_scalar(tag, scalar_value, global_step=None, )：
                    将我们所需要的数据保存在文件里面供可视化使用
                    scalar_value（浮点型或字符串）：y轴数据（步数）
                '''
                batch_loss.append(loss.item())
                '''
                    batch_loss = []
                    在for 循环下一个epoch，append一次

                '''
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            '''
                走完所有epoch
                epoch_loss = []
                对loss求平均
            '''
            '''
                state_dict变量存放训练过程中需要学习的权重和偏执系数
                state_dict作为python的字典对象将每一层的参数映射成tensor张量
                需要注意的是torch.nn.Module模块中的state_dict只包含卷积层和全连接层的参数
            '''
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        '''
            返回推断准确率和损失
        '''
        model.eval()
        '''
            1.告诉我们的网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
            使用PyTorch进行训练和测试时一定注意要把实例化的
            model指定train/eval，eval（）时，框架会自动把
            BN和DropOut固定住，不会取平均，而是用训练好的值，
            不然的话，一旦test的batch_size过小，很容易就会
            被BN层导致生成图片颜色失真极大
        '''
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            '''

                batch_idx
                (images, labels)
                enumerate(self.trainloader)会将训练集数据一个
                batch一个batch地取出来用来训练
                使用enumerate进行dataloader中的数据读取用于
                神经网络的训练是第一种数据读取方法，其基本形式
                即为for index， item in enumerate(dataloader['train'])，
                其中item中0为数据，1为label.

            '''
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference推断
            outputs = model(images)

            batch_loss = self.criterion(outputs, labels)
            '''
                要这样写
                crossentropyloss=nn.CrossEntropyLoss()
                crossentropyloss_output=crossentropyloss(x_input,y_target)
                print('crossentropyloss_output:\n',crossentropyloss_output)


                bool value of Tensor with more than one value is ambiguous
                不能直接这样写：
                xx = nn.CrossEntropyLoss(x_input, y_target)
                print(xx)

            '''
            loss += batch_loss.item()

            # Prediction
            '''
                预测过程
            '''
            _, pred_labels = torch.max(outputs, 1)
            '''
                _是每行的最大值
                pred_labels表示每行最大值的索引
                但是为什么命名为pred_labels(预测标签)

                torch.max(input, dim) 
                input是softmax函数输出的一个tensor
                dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值

                函数会返回两个tensor，
                第一个tensor是每行的最大值；
                第二个tensor是每行最大值的索引
            '''
            pred_labels = pred_labels.view(-1)
            '''
                tensor([2, 2]).view(-1)结果不变
                tensor([[1, 1, 1]]).view(-1)结果为tensor([1, 1, 1])

                b=torch.Tensor([[[1,2,3],[4,5,6]]])
                它的结构是
                [1,4]
                [2,5]
                [3,6]
                a=torch.Tensor([[1,2,3],[4,5,6]])
                它的结构是
                [1,2,3]
                [4,5,6]
                b=torch.Tensor([[[1,2,3]]])
                它的结构是
                [1]
                [2]
                [3]


            '''
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            # corrent存储预测标签与真实标签有多少个匹配的
            '''
            torch.eq:
                比较两个tensor对应位置数字是否相同，相同为1，否则为0
                pred_label=torch.Tensor([1,2,3])
                labels=torch.Tensor([1,0,3])
                torch.eq(pred_label, labels):
                    tensor([ True, False,  True])
                torch.sum(torch.eq(pred_label, labels):
                    tensor(2)
            torch.sum:
                就是将所有值相加，但是得到的仍然是一个tensor
                tensor(2)
            tensor.item():
                label=torch.Tensor([2])
                print(label.item())
                tensor中只有一个元素可以这样用item()
            '''
            total += len(labels)
            '''
                total多少行？
            '''

        accuracy = correct / total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    # 返回测试的精度和损失
    """
        def inference(self, model):
        test推断函数和原推断函数还是不同的
        首先数据集就不同
        原推断函数是：self.testloader
            def train_val_test(self, dataset, idxs):
            用的是这个函数返回的testloader

        test推断函数：数据集是test_dataset

        Returns the test accuracy and loss.
    """

    model.eval()
    '''
       1.告诉我们的网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
       使用PyTorch进行训练和测试时一定注意要把实例化的
       model指定train/eval，eval（）时，框架会自动把
       BN和DropOut固定住，不会取平均，而是用训练好的值，
       不然的话，一旦test的batch_size过小，很容易就会
    '''
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)

    '''
        默认标准设置为NLL损失函数
        类似：net = net.to(device)
        CrossEntropyLoss()=log_softmax() + NLLLoss() 
        softmax(x)+log(x)+nn.NLLLoss====>nn.CrossEntropyLoss
        其中softmax函数又称为归一化指数函数，它可以把一个多维向量
        压缩在（0，1）之间，并且它们的和为1
    '''
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)
    '''
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
    '''

    # 上面三行是与原推断函数不同的代码
    # 还有下面for循环中原推断函数是enumerate(self.testloader)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        '''

            batch_idx
            (images, labels)
            enumerate(self.trainloader)会将训练集数据一个
            batch一个batch地取出来用来训练
            使用enumerate进行dataloader中的数据读取用于
            神经网络的训练是第一种数据读取方法，其基本形式
            即为for index， item in enumerate(dataloader['train'])，
            其中item中0为数据，1为label.
        '''
        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        '''
            要这样写
            crossentropyloss=nn.CrossEntropyLoss()
            crossentropyloss_output=crossentropyloss(x_input,y_target)
            print('crossentropyloss_output:\n',crossentropyloss_output)


            bool value of Tensor with more than one value is ambiguous
            不能直接这样写：
            xx = nn.CrossEntropyLoss(x_input, y_target)
            print(xx)

        '''
        loss += batch_loss.item()

        # Prediction
        '''
            预测
        '''
        _, pred_labels = torch.max(outputs, 1)
        '''
             _是每行的最大值
             pred_labels表示每行最大值的索引
             但是为什么命名为pred_labels(预测标签)

             torch.max(input, dim) 
             input是softmax函数输出的一个tensor
             dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值

             函数会返回两个tensor，
             第一个tensor是每行的最大值；
             第二个tensor是每行最大值的索引
         '''
        pred_labels = pred_labels.view(-1)
        '''
            tensor([2, 2]).view(-1)结果不变
            tensor([[1, 1, 1]]).view(-1)结果为tensor([1, 1, 1])

            b=torch.Tensor([[[1,2,3],[4,5,6]]])
            它的结构是
            [1,4]
            [2,5]
            [3,6]
            a=torch.Tensor([[1,2,3],[4,5,6]])
            它的结构是
            [1,2,3]
            [4,5,6]
            b=torch.Tensor([[[1,2,3]]])
            它的结构是
            [1]
            [2]
            [3]


        '''
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        # corrent存储预测标签与真实标签有多少个匹配的
        '''
        torch.eq:
            比较两个tensor对应位置数字是否相同，相同为1，否则为0
            pred_label=torch.Tensor([1,2,3])
            labels=torch.Tensor([1,0,3])
            torch.eq(pred_label, labels):
                tensor([ True, False,  True])
            torch.sum(torch.eq(pred_label, labels):
                tensor(2)
        torch.sum:
            就是将所有值相加，但是得到的仍然是一个tensor
            tensor(2)
        tensor.item():
            label=torch.Tensor([2])
            print(label.item())
            tensor中只有一个元素可以这样用item()
        '''
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss


