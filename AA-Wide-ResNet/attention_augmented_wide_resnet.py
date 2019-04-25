import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np

from attention_augmented_conv import AugmentedConv


# reference, Thank you!!
# https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, v=0.2, k=2, Nh=4):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = AugmentedConv(in_planes, planes, kernel_size=3, dk=k * planes, dv=int(v * planes), Nh=Nh, relative=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = AugmentedConv(planes, planes, kernel_size=3, dk=k * planes, dv=int(v * planes), Nh=Nh, stride=stride, relative=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                AugmentedConv(in_planes, planes, kernel_size=3, dk=k * planes, dv=int(v * planes), Nh=Nh, relative=True, stride=stride),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        short = self.shortcut(x)
        out += short

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 20

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        dv_v = 0.2
        dk_k = 2
        Nh = 4

        print('| Wide-Resnet %dx%d' % (depth, k))
        n_Stages = [20, 20 * k, 40 * k, 40 * k]

        self.conv1 = AugmentedConv(3, n_Stages[0], kernel_size=3, dk=dk_k * n_Stages[0], dv=int(dv_v * n_Stages[0]), Nh=Nh, relative=True)
        self.layer1 = nn.Sequential(
            self._wide_layer(wide_basic, n_Stages[1], n, dropout_rate, stride=1),
        )
        self.layer2 = nn.Sequential(
            self._wide_layer(wide_basic, n_Stages[2], n, dropout_rate, stride=2),
        )
        self.layer3 = nn.Sequential(
            self._wide_layer(wide_basic, n_Stages[3], n, dropout_rate, stride=2),
        )
        self.bn1 = nn.BatchNorm2d(n_Stages[3], momentum=0.9)
        self.linear = nn.Linear(n_Stages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
