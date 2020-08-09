import torch
import torchvision
import logging
from torch import nn


def get_logger():
    return logging.getLogger("Resnet18")


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self._logger = get_logger()
        self.conv1 = nn.Conv2d(in_channel, out_channel, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        return x


class ResNet(torch.nn.Module):

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self._logger = get_logger()
        self._logger.info(f"Resenet Initialization: blocks {block}, layers {layers}")
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self._logger.debug(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        self._logger.debug(self.in_channels)
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )
        layers = list()
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)


resnet_spec = {"18": (BasicBlock, [2, 2, 2, 2]),
               "34": (BasicBlock, [3, 4, 6, 3])}


def get_resnet(num_layers):
    block, layers = resnet_spec[num_layers]
    resnet = ResNet(block, layers)
    return resnet