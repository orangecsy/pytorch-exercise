'''
pytorch学习笔记
基础练习
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 打印版本
# print(torch.__version__)

# 未初始化的矩阵
# x = torch.empty(3, 2)
# print(x)

# 随机初始化
# x = torch.rand(5, 3)
# print(x)

# 类型为long的全0填充矩阵
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)

# 已有变量转为tensor
# x = torch.tensor([3.1, 2])
# print(x)
# 张量维数
# print(x.size())

# 加法
# x = torch.tensor([3, 2], dtype=torch.float)
# y = torch.tensor([1, 3], dtype=torch.float)
# print(x + y)
# print(torch.add(x, y))

# res = torch.empty(2)
# torch.add(x, y, out=res)
# print(res)

# print(y.add(x))
# print(y)
# 操作后加_会替换点号前变量
# print(y.add_(x))
# print(y)

# 索引
# x = torch.tensor([
#     [1, 3],
#     [4, 6]
# ])
# print(x[:, 1])

# x = torch.randn(2, 3)
# y = x.view(6)
# -1从其他纬度推断
# z = x.view(3, -1)
# print(x.size(), y.size(), z.size())

# 只有一个元素的张量，使用.item()来得到Python数据类型的数值
# x = torch.randn(1)
# print(x)
# print(x.item())

# 转numpy
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# numpy和tensor共用内存
# a.add_(1)
# print(a)
# print(b)

# a = np.ones(3)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)
# print(a)
# print(b)

# to方法移动变量到设备
# x = torch.tensor([1, 3])
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     y = torch.ones_like(x, device=device)
#     x = x.to(device)
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))

# 追踪张量操作 => 设置 .requires_grad 为 True
# 可包装在with torch.no_grad()：中
# 调用 .detach() 方法禁止追踪

'''
神经网络的典型训练过程：
1、定义包含一些可学习的参数(或者叫权重)神经网络模型；
2、在数据集上迭代；
3、通过神经网络处理输入；
4、计算损失(输出结果和正确值的差值大小)；
5、将梯度反向传播回网络的参数；
6、更新网络的参数，主要使用如下简单的更新原则：  weight = weight - learning_rate * gradient
'''

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input, 6 output, 5*5 square convolution
#         # nn.Conv2d 接受一个4维的张量
#         # 每一维分别是 sSamples * nChannels * Height * Width（样本数*通道数*高*宽）
#         # 单个样本，需使用 input.unsqueeze(0) 来添加其它的维数
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
    
#     # backward 函数（用来计算梯度）会被autograd自动创建
#     # forward 函数中使用任何针对Tensor的操作
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
#     def num_flat_features(self, x):
#         size = x.size()[1:]
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features

# net = Net()
# print(net)
# net.parameters() 返回可被学习的参数（权重）列表和值
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())

# torch.Tensor：调用 backward()自动计算梯度
# nn.Module：神经网络模块，封装参数
# 损失函数接受一对 (output, target) 作为输入，计算一个值来估计网络的输出和目标值相差多少
# net = Net()
# input = torch.randn(1, 1, 32, 32)
# output = net(input)
# target = torch.randn(10)
# # print(target.size())
# target = target.view(1, -1)
# # print(target.size())
# criterion = nn.MSELoss()
# loss = criterion(output, target)
# print(loss)
# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0])
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) 

# 优化器 随机梯度下降SGD 权重更新规则
# net = Net()
# optimizer = optim.SGD(net.parameters(), lr=0.01)
# optimizer.zero_grad()
# input = torch.randn(1, 1, 32, 32)
# output = net(input)
# criterion = nn.MSELoss()
# target = torch.randn(10)
# target = target.view(1, -1)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()
