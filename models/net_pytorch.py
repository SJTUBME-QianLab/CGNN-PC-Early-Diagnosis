#coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import random
import os


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 使用简化的6层VGG模型进行patch层面分类和特征提取
class VGG_liuyedao_ori_cnn_CEL(nn.Module):
    def __init__(self):
        super(VGG_liuyedao_ori_cnn_CEL,self).__init__()
        self.conv1a = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 灰色图像通道数是1
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
        )
        self.conv1b = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 使用了一个2×2的最大池化
        )
        self.conv2a = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3a = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv3b = nn.Sequential(
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*6*6,32),
            nn.ReLU(),)
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.fc2 = nn.Sequential(
            nn.Linear(32,32),
            nn.ReLU(),)
        self.fc3 = nn.Sequential(
            nn.Linear(32,2),
            nn.Sigmoid(),)

    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  # feature prob

    # 输出patch特征，用于后续图节点特征的构建
    def feature(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x1 = self.fc2(x)  # feature
        return x1
