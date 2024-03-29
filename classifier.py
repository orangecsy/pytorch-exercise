'''
pytorch学习笔记
简单分类器
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np

# 是否支持gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

'''
训练图像分类器
1、使用torchvision加载和归一化CIFAR10训练集和测试集
2、定义一个卷积神经网络
3、定义损失函数
4、在训练集上训练网络
5、在测试集上测试网络
'''

# 1. 读取和归一化 CIFAR10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))

# 报错 RuntimeError: 
# An attempt has been made to start a new process before the current process has finished its bootstrapping phase
# 解决方法：放到main函数内
# if __name__ == '__main__':
#     # 获取随机数据
#     dataiter = iter(trainloader)
#     images, labels = dataiter.next()

#     # 展示图像
#     imshow(torchvision.utils.make_grid(images))

#     # 显示图像标签
#     print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#     plt.show()

# 2. 定义一个卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # dropout 提升准确率
        # self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
# L2正则化 optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.01)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    # 4. 训练网路
    for epoch in range(2):  # 多批次循环
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度置0
            optimizer.zero_grad()

            # 正向传播，反向传播，优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            running_loss += loss.item()
            if i % 2000 == 1999:    # 每2000批次打印一次
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    # 显示图片
    # imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # 训练结果
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    # 看下网络识别训练集的情况
    # plt.show()

    # 测试集的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # 分类统计准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
