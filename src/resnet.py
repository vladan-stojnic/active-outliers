"""ResNet in PyTorch.

Reference
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
"""


import copy
import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models import resnet18 as resnet18_imagenet
from torchvision.models import resnet50 as resnet50_imagenet
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        channels=3,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.embDim = 8 * self.in_planes * block.expansion

        self.conv1 = nn.Conv2d(
            channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.num_classes = num_classes

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _get_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = F.avg_pool2d(out, 4)
        e = out.view(out.size(0), -1)

        return e

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                e = self._get_features(x)
        else:
            e = self._get_features(x)

        out = self.linear(e)

        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim


def ResNet18(num_classes=10, channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, channels)


def ResNet34(num_classes=10, channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, channels)


def ResNet50(num_classes=10, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)


def ResNet101(num_classes=10, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)


def ResNet152(num_classes=10, channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)


class ResNet18ImageNet(nn.Module):
    def __init__(self, num_classes=10, channels=3, pretrained=False):
        super().__init__()
        weights = None
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18_imagenet(weights=weights)
        self.backbone.fc = nn.Identity()

        self.embDim = 512

        self.classifier = nn.Linear(self.embDim, num_classes)

    def _get_features(self, x):
        out = self.backbone(x)

        return out

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                e = self._get_features(x)
        else:
            e = self._get_features(x)

        out = self.classifier(e)

        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim


class SIMCLRModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.simclr_layer = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.simclr_layer(x)


class ResNet18ImageNetCSI(nn.Module):
    def __init__(self, num_classes=10, channels=3, pretrained=False):
        super().__init__()
        weights = None
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18_imagenet(weights=weights)
        self.backbone.fc = nn.Identity()

        self.embDim = 128

        self.simclr = SIMCLRModule(512, 128)

    def _get_features(self, x):
        out = self.backbone(x)

        return out

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                e = self._get_features(x)
        else:
            e = self._get_features(x)

        out = self.simclr(e)

        if last:
            return out, out
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim


class ResNet18CSI(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super().__init__()
        self.backbone = ResNet18(num_classes=num_classes, channels=channels)
        self.embDim = 128
        self.simclr = SIMCLRModule(512, 128)

    def _get_features(self, x):
        _, out = self.backbone.forward(x, last=True)

        return out

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                e = self._get_features(x)
        else:
            e = self._get_features(x)

        out = self.simclr(e)

        if last:
            return out, out
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim


class ResNet50ImageNet(nn.Module):
    def __init__(self, num_classes=10, channels=3, pretrained=False):
        super().__init__()
        weights = None
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        self.backbone = resnet50_imagenet(weights=weights)
        self.backbone.fc = nn.Identity()

        self.embDim = 2048

        self.classifier = nn.Linear(self.embDim, num_classes)

    def _get_features(self, x):
        out = self.backbone(x)

        return out

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                e = self._get_features(x)
        else:
            e = self._get_features(x)

        out = self.classifier(e)

        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim


class MLPDetector(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x, last=False, freeze=False):
        out = self.fc(x)
        if last:
            return out, out
        else:
            return out


class EnsembleModel(nn.Module):
    def __init__(self, number_of_models=10, original_model=None, cfg=None):
        super().__init__()
        self.models = [original_model]
        self.models.extend(
            [copy.deepcopy(original_model) for _ in range(number_of_models - 1)]
        )
        self.active = 0
        self.cfg = cfg
        self.num_of_models = number_of_models

    def get_model(self, idx):
        (self.models[self.active]).to("cpu")
        (self.models[self.active]).requires_grad_(False)
        self.active = idx
        (self.models[self.active]).to(self.cfg.device)
        (self.models[self.active]).requires_grad_(True)
        return self.models[self.active]
