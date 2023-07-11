#模型

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''
        MLP模型
        通用代码
    '''

    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        '''
            定义函数
            torch.Linear()设置网络中的全连接层
        '''
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
            不能直接这样写：
                nn.ReLU(input)
            应该这样写：
                relu = nn.ReLU()
                input = relu(input)

        '''
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        '''
        1.torch.randn()
            torch.randn是标准正态分布的随机变量
            # 假定输入的图像形状为[64,64,3]
            input = torch.randn(1,3,4,4)
            表示
            第一个参数：batch_size为1
            第二个参数：3表示通道图片为RGB彩色图片
            第三个参数和第四个参数：
                图片尺寸为4*4 

            x.shape[1]*x.shape[-2]*x.shape[-1]
            表示3*4*4
            输入图像为[4,4,3]
        2.x.view(-1,)
            例如一个长度的16向量x，

            x.view(-1, 4)等价于x.view(4, 4)

            x.view(-1, 2)等价于x.view(8，2)
            长度为48的向量
            所以对于上面的x=x.view(-1,48)
            等价于x.view(1,48)

        '''
        x = self.layer_input(x)
        '''
            铺平送入网络
            layer_input = nn.Linear(dim_in, dim_hidden)
        '''
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        '''
            过程解析在test.py
        '''
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        '''
            torch.nn.Conv2d
            (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            in_channels：输入图像通道数
            out_channels：卷积产生的通道数
            kernel_size：卷积核尺寸，可以设为1个int型数或者一个(int, int)型的元组	
            stride：卷积步长，默认为1。可以设为1个int型数或者一个(int, int)型的元组。

            x = torch.randn(3,1,4,4)
                x[ batch_size, channels, height_1, width_1 ]
                batch_size，一个batch中样本的个数 3
                channels，通道数，也就是当前层的深度 1
                height_1， 图片的高 4
                width_1， 图片的宽 4

            conv = torch.nn.Conv2d(1,4,(2,2))
                ****************************
                channels，通道数，和上面保持一致，也就是当前层的深度 1
                output ，输出的深度 4【需要4个filter】
                height_2，卷积核的高 2
                width_2，卷积核的宽 2
            res = conv(x)
                res[ batch_size,output, height_3, width_3 ]
                batch_size,，一个batch中样例的个数，同上 3
                output， 输出的深度 4
                height_3， 卷积结果的高度 3
                width_3，卷积结果的宽度 3
        '''
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)
        '''
            经过卷积之后torch.Tensor维度为1*320
            torch.Tensor

        '''

        self.fc2 = nn.Linear(50, args.num_classes)
        '''
            经过第二个全连接层输出维度1*10
        '''

    '''
        # 1为图片通道数
        # 本地batch_size=10
        x = torch.randn(1, 1, 28, 28)
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        # print(x.shape):(10, 1, 28, 28)->torch.Size([10, 784])
        # (1, 1, 28, 28)->torch.Size([1, 784])
        # 1为输入图像通道数，10为输出图像通道数
        # 需要10个卷积核才会输出10个通道的结果
        conv1 = nn.Conv2d(1, 10, kernel_size=5)  # num_channels通道为1，无色图

        # 10为输入图像通道数，20为输出图像通道数
        conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # dropout
        conv2_drop = nn.Dropout2d()

        # 第一个全连接层, 320为输入维度,50为全连接层输出维度
        fc1 = nn.Linear(320, 50)

        # 第二个全连接层
        fc2 = nn.Linear(50, 10)  # args.num_classes

        # 经过全连接等过程 1*320 # 320*50 = 1*50, 1*50 # 50*10 = 1*10
    '''

    def forward(self, x):
        '''
            x=x = torch.randn(1, 1, 28, 28)
            一次一个样本图片
            batch_size=1

            28×28经过conv1->24×24经过max_pool->12×12
            12×12经过conv2->8×8经过max_pool->4×4
            这时通道数20
            进入全连接层之前为：20*4*4
            铺平x为1*320
            经过第一层全连接层(320*50) x为1*50
            经过第二层全连接层(50*10) x为1*10
            最后经过softmax再取对数log

        '''

        '''
            F.dropout实际上是torch.nn.functional.dropout
        '''
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        ''''
            维度减半,步长stride为2
        '''
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        '''
            28×28经过conv1->24×24经过max_pool->12×12
            12×12经过conv2->8×8经过max_pool->4×4
            这时通道数20
        '''
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        '''
            torch.Tensor铺平为1×320
        '''
        x = F.relu(self.fc1(x))
        '''
            第一个全连接层, 320为输入维度,50为全连接层输出维度
            输出的x为1×50
        '''
        x = F.dropout(x, training=self.training)
        '''
            当training 是真的时候，才会将一部分元素置为0，
            其他元素会乘以 scale 1/(1-p)

            F.dropout(input,p=0.5, training = True)
            默认有一半被dropout即p=0.5

        '''

        x = self.fc2(x)
        '''
            第二个全连接层, 50为输入维度,10为全连接层输出维度
            输出的x为1×10
        '''
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        '''
            1.torch.nn.MaxPool2d和torch.nn.functional.max_pool2d
            nn.MaxPool2d(2))写在nn.Sequential()里面
            相当于torch.nn.MaxPool2d()
            2.torch.nn.MaxPool2d在自己的forward()方法中调用了
            torch.nn.functional.max_pool2d。
            两者本质上是一样的

            3.import torch.nn.functional as F
            torch.nn.functional.max_pool2d作为函数可以直接调用
            传入参数（input（四个维度(***4D***)的输入张量）, kernel_size（卷积核尺寸）
            stride（步幅）,padding（填充）等等
            F.max_pool2d(self.conv1(x), 2)
            2为kernel_size

            4.torch.nn.MaxPool2d，要先实例化，并在自身类的
            forward调用了torch.nn.functional.max_pool2d函数。

            5.一下
            我们通常会先pooling再relu
            但relu和maxpooling，把它们当算子来看的话，
            都是对顺序不敏感的，也就是可交换的。
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))  # from torch import nn
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    '''

        6.
        torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, 
        track_running_stats=True, device=None, dtype=None)		
        作用：对4D输入（具有附加通道尺寸的2D输入的小批量）应用批量归一化
        1e-05==0.00001
        (1)输入的数据格式：batch_size ，num_features ， height ， width
        num_features为channel数
        (2)affine一个布尔值，当设置为True时，此模块具有可学习的仿射参数。
        γ(gamma) 和 β(beta) （可学习的仿射变换参数） 
        默认值：True
        (3)进行模型训练之前，需对数据做归一化处理，使其分布一致。
        在深度神经网络训练过程中，通常一次训练是一个batch，而非全体数据。
        每个batch具有不同的分布产生了internal covarivate shift
        问题——在训练过程中，数据分布会发生变化，对下一层网络的学习
        带来困难。Batch Normalization强行将数据拉回到均值为0，
        方差为1的正态分布上，
        一方面使得数据分布一致，
        另一方面避免梯度消失。




        x = torch.randn(1, 1, 28, 28)


    '''

    def forward(self, x):
        '''
            1*1*28*28经过padding=2
            1*1*32*32经过卷积层
            1*16*28*28经过pooling
            1*16*14*14经过padding=2
            1*16*18*18经过卷积层
            1*32*14*14经过pooling
            1*32*7*7

        '''
        out = self.layer1(x)
        out = self.layer2(out)
        '''
            经过卷积层：1*32*7*7
            1*448
        '''
        out = out.view(out.size(0), -1)
        '''
            16
            x.view(-1, 4)等价于x.view(4, 4)

            x.view(-1, 2)等价于x.view(8，2)
            长度为48的向量
            所以对于上面的x=x.view(-1,48)
            等价于x.view(1,48)

            7*7*32
            out.view(out.size(0), -1)
            等价于：
            out.view(out.size(0), 1)
            out.view(1, 448)

        '''
        out = self.fc(out)
        '''
            上面定义的全连接层为：
            nn.Linear(448, 10)
            经过全连接层之后tensor变为
            1*10
            [,,,,,,,,,]
        '''

        return out


'''
     CIFAR-10图片尺寸是32 x 32，
     稍大于MNIST的28 x 28 CIFAR-10中物体的比例和特征不尽相同，
     噪声大，识别难度较MNIST高
     MNIST是黑白
     Cifar是彩色
     通道为RGB==3
'''


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        '''
            (1)nn.Conv2d(3, 6, 5)
            卷积核通道为3
            输出通道为6
            卷积核大小5*5
            (2)nn.MaxPool2d(2, 2)
            池化卷积核大小2*2
            步长为2
        '''
        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)  # 下面没少吗？

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        '''
             CIFAR-10数据集包含60000张 32x32的彩色图片，共分为10种类别，每种类别6000张
        '''
        '''
            (1)卷积公式
                1 + int((H-F+2p)/s)
            (2)池化公式
                1 + int((H-H_filter)/s)
            1*3*32*32经过卷积层  32-5+1，分母为步长=1
            1*6*28*28经过pooling (28-2)/2 + 1 = 14   x = self.pool(F.relu(self.conv1(x)))
            1*6*14*14经过卷积层
            1*16*10*10经过pooling
            1*16*5*5                x = self.pool(F.relu(self.conv2(x)))

        '''
        x = self.pool(F.relu(self.conv1(x)))
        # print('0000000000000000000000000000000')
        # print(x.size(0))
        # print(x.size(1))
        # print(x.size(2))
        # print(x.size(3))
        x = self.pool(F.relu(self.conv2(x)))
        # print('0000000000000000000000000000000')
        # print(x.size(0))
        # print(x.size(1))
        # print(x.size(2))
        # print(x.size(3))

        x = x.view(-1, 16 * 5 * 5)
        '''
            x = x.view(-1, 16 * 5 * 5)
            相当于：
            x = x.view(10, 16 * 5 * 5)
            这里x.size(0)=10
        '''

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        '''
            该实验下输入的是10*3*32*32
            batch_size=10
            一下子放10个样本进入神经网络训练
            经过卷积之后为：
            (10, 400)
            经过第一层全连接层；(400, 120)
            (10, 120)
            经过第二层全连接层；(120, 84)
            (10, 84)
            经过第三层全连接层；(84, 10)
            (10, 10)
        (10,10)长什么样？
        tensor([ [],[],[],[],[],[],[],[],[],[] ])  

        '''
        return F.log_softmax(x, dim=1)





