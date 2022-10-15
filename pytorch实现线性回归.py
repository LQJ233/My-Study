import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import time

# 1. 定义数据
x = torch.rand([50, 1])#x，y的形状为50行1列
y = x * 3 + 0.8


# 2 .定义模型
class Lr(nn.Module):#nn.Modul 是torch.nn提供的一个类，是pytorch中我们自定义网络的一个基类，在这个类中定义了很多有用的方法
    def __init__(self):
        super(Lr, self).__init__()#super函数继承父类init的参数
        self.linear = nn.Linear(1, 1)#nn.Linear为torch预定义好的线性模型，也被称为全链接层，传入的参数为输入的数量，输出的数量，这里因为输入输出均为1列

    def forward(self, x):#前向传播
        out = self.linear(x)
        return out
    #一般我们都在__init__方法中定义模型一些所需要的层，
    # 而在forward方法中，将传入模型的数据通过__init__方法中定义的层，还能在里面加一些激活函数等，也就是对数据进行处理，最后将这个值返回。

# 3. 实例化模型，loss，和优化器
"""正常使用"""
#model = Lr()
# 损失函数
#criterion = nn.MSELoss()
# 优化器
#optimizer = optim.SGD(model.parameters(), lr=1e-3)

"""GPU加速"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#判断是否能用gpu
x, y = x.to(device), y.to(device)

model = Lr().to(device)#实例化模型
criterion = nn.MSELoss()#实例化损失函数
#均方误差：nn.MSELoss(),常用于回归问题
#交叉熵损失：nn.CrossEntropyLoss()，常用于分类问题
optimizer = optim.SGD(model.parameters(), lr=1e-3)#实例化优化器
#SGD随机梯度下降，优化器都是optim.什么什么
#参数可以使用model.parameters()来获取，获取模型中所有requires_grad=True的参数

"""优化器使用步骤"""
#optimizer = optim.SGD(model.parameters(), lr=1e-3) #1. 实例化，1e-3也可以写成0.001
#optimizer.zero_grad() #2. 梯度置为0
#loss.backward() #3. 计算梯度
#optimizer.step()  #4. 更新参数的值





# 4. 训练模型
for i in range(30000):#epoch,训练30000个epoch（周期）
    out = model(x)#前向传播计算预测值
    loss = criterion(y, out)#调用损失函数传入真实值和预测值，得到损失结果

    optimizer.zero_grad()#当前循环参数梯度置为0
    loss.backward()#计算梯度
    optimizer.step()#更新参数的值
    if (i + 1) % 20 == 0:#打印损，画图
        print('Epoch[{}/{}], loss: {:.6f}'.format(i, 30000, loss.data))
"""通常训练步骤"""
# model = Lr() #1. 实例化模型
# criterion = nn.MSELoss() #2. 实例化损失函数
# optimizer = optim.SGD(model.parameters(), lr=1e-3) #3. 实例化优化器类
# for i in range(100):
#     y_predict = model(x_true) #4. 向前计算预测值
#     loss = criterion(y_true,y_predict) #5. 调用损失函数传入真实值和预测值，得到损失结果
#     optimizer.zero_grad() #5. 当前循环参数梯度置为0
#     loss.backward() #6. 计算梯度
#     optimizer.step()  #7. 更新参数的值


# 5. 模型评估
model.eval()#表示设置模型为评估模式，即预测模式
predict = model(x)
predict = predict.cpu().detach().numpy()  # 转化为numpy数组
plt.scatter(x.cpu().data.numpy(), y.cpu().data.numpy(), c="r")
plt.plot(x.cpu().data.numpy(), predict)
plt.show()
#model.train(mode=True) 表示设置模型为训练模式


# 判断GPU是否可用torch.cuda.is_available()
# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# >>device(type='cuda', index=0)  #使用gpu
# >>device(type='cpu') #使用cpu


# 把模型参数和input数据转化为cuda的支持类型
# model.to(device)
# x_true.to(device)

# 在GPU上计算结果也为cuda的数据类型，需要转化为numpy或者torch的cpu的tensor类型
# predict = predict.cpu().detach().numpy()
# detach()的效果和data的相似，但是detach()是深拷贝，data是取值，是浅拷贝
