import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_augmented_conv import AugmentedConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, shape, stride=1, v=0.2, k=2, Nh=4):
        super(wide_basic, self).__init__()
        if stride == 2:
            original_shape = shape * 2
        else:
            original_shape = shape

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = AugmentedConv(in_planes, planes, kernel_size=3, dk=k * planes, dv=int(v * planes), Nh=Nh, relative=True, shape=original_shape)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = AugmentedConv(planes, planes, kernel_size=3, dk=k * planes, dv=int(v * planes), Nh=Nh, stride=stride, relative=True, shape=shape)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                AugmentedConv(in_planes, planes, kernel_size=3, dk=k * planes, dv=int(v * planes), Nh=Nh, relative=True, stride=stride, shape=shape),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        short = self.shortcut(x)
        out += short

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, shape):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 20
        self.shape = shape

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        dv_v = 0.2
        dk_k = 2
        Nh = 4

        print('| Wide-Resnet %dx%d' % (depth, k))
        n_Stages = [20, 20 * k, 40 * k, 60 * k]

        self.conv1 = AugmentedConv(in_channels=3, out_channels=n_Stages[0], kernel_size=3, dk=dk_k * n_Stages[0], dv=int(dv_v * n_Stages[0]), shape=shape, Nh=Nh, relative=True)
        self.layer1 = nn.Sequential(
            self._wide_layer(wide_basic, n_Stages[1], n, dropout_rate, stride=1, shape=shape),
        )
        self.layer2 = nn.Sequential(
            self._wide_layer(wide_basic, n_Stages[2], n, dropout_rate, stride=2, shape=shape // 2),
        )
        self.layer3 = nn.Sequential(
            self._wide_layer(wide_basic, n_Stages[3], n, dropout_rate, stride=2, shape=shape // 4),
        )
        self.bn1 = nn.BatchNorm2d(n_Stages[3], momentum=0.9)
        self.linear = nn.Linear(n_Stages[3], num_classes)

        self.apply(_weights_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, shape):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate=dropout_rate, stride=stride, shape=shape))
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

# tmp = torch.randn((4, 3, 32, 32))
# model = Wide_ResNet(28, 10, 0.3, num_classes=100, shape=32)
# print(model(tmp).shape)
#
# for name, param in model.named_parameters():
#     print('parameter name: ', name)
