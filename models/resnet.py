from torch import nn
import torch.nn.functional as F

from models.common import BaseModel, CustomSequential

# Resnet implementation based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py


class BasicBlock(BaseModel):
    expansion = 1

    def __init__(self, ConvLayer, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__(ConvLayer)
        self.conv1 = self.ConvLayer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self.ConvLayer(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = CustomSequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = CustomSequential(
                self.ConvLayer(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, temperature):
        out = F.relu(self.bn1(self.conv1(x, temperature)))
        out = self.bn2(self.conv2(out, temperature))
        out += self.shortcut(x, temperature)
        out = F.relu(out)
        return out


class ResNet(BaseModel):
    def __init__(self, ConvLayer, block, num_blocks, num_classes=200):
        super(ResNet, self).__init__(ConvLayer)
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # First convolution non-dynamic
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return CustomSequential(*layers)

    def forward(self, x, temperature):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out, temperature)
        out = self.layer2(out, temperature)
        out = self.layer3(out, temperature)
        out = self.layer4(out, temperature)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)
        return out


def ResNet10(ConvLayer):
    return ResNet(ConvLayer, BasicBlock, [1, 1, 1, 1])


def ResNet18(ConvLayer):
    return ResNet(ConvLayer, BasicBlock, [2, 2, 2, 2])