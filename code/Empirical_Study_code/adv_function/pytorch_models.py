import torch.nn as nn
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from torch.nn.init import xavier_normal, zeros_


class CustomDataset(data.Dataset):#需要继承data.Dataset
    def __init__(self, X, Y):
        self.images = np.transpose(X, [0, 3, 1, 2]).astype('double')
        self.labels = Y.reshape(Y.shape[0], 1).astype('long')

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_normal(m.weight.data)
        zeros_(m.bias.data)


def weights_m_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal(m.weight.data)
            nn.init.xavier_normal(m.weight.data)
            nn.init.kaiming_normal(m.weight.data)  # 卷积层参数初始化
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_()  # 全连接层参数初始化


class ConvNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),#32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(32*9*9, num_classes)
        # self.fc1 = nn.Linear(32*9*9, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.fc3(out)
        return out


def evalute(model, test_loader, device):
    m_model = model.eval()
    tot_acc_num = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            acc_num = int(torch.sum(torch.argmax(m_model(images), axis=1) == labels.squeeze()).cpu())
            tot_acc_num += acc_num
    acc = tot_acc_num / len(test_loader.dataset)
    return acc