"""ResNet backbones used by BUAL."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
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
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
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
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bias=False,
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
        return F.relu(out)


class ResNet(nn.Module):
    """Dual-path ResNet used for spatial and wavelet-frequency branches."""

    def __init__(self, block, num_blocks, num_classes: int = 10, frequency: bool = False):
        super().__init__()
        self.in_planes = 64
        self.in_planes_frequency = 64
        self.frequency = frequency

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.conv_frequency = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_frequency = nn.BatchNorm2d(64)
        self.layer1_frequency = self._make_frequency_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_frequency = self._make_frequency_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_frequency = self._make_frequency_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_frequency = self._make_frequency_layer(block, 512, num_blocks[3], stride=2)

        embedding_dim = 512 * block.expansion
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.frequency_classifier = nn.Linear(embedding_dim, num_classes)
        self.consistency_head = nn.Linear(embedding_dim, num_classes)
        self.frequency_consistency_head = nn.Linear(embedding_dim, num_classes)

        self.dwt = DWTForward(J=1, wave="sym17", mode="symmetric")
        self.register_buffer("H_matrix", torch.randn(12, 3, 3, 3))

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_frequency_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes_frequency, planes, block_stride))
            self.in_planes_frequency = planes * block.expansion
        return nn.Sequential(*layers)

    def random_h_transform(self, x):
        return F.conv2d(x, self.H_matrix, stride=1, padding=1)

    def forward(self, x):
        if self.frequency:
            return self._forward_frequency(x)
        return self._forward_spatial(x)

    def _forward_spatial(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        embedding = F.avg_pool2d(out4, 4).view(out4.size(0), -1)
        logits = self.classifier(embedding)
        consistency_logits = self.consistency_head(embedding)
        return logits, embedding, consistency_logits, [out1, out2, out3, out4]

    def _forward_frequency(self, x):
        x = self.wavelets(x)
        x = x[:, 0:3, :, :] + x[:, 3:6, :, :] + x[:, 6:9, :, :] + x[:, 9:12, :, :]
        out = F.relu(self.bn_frequency(self.conv_frequency(x)))
        out1 = self.layer1_frequency(out)
        out2 = self.layer2_frequency(out1)
        out3 = self.layer3_frequency(out2)
        out4 = self.layer4_frequency(out3)
        embedding = F.avg_pool2d(out4, 4).view(out4.size(0), -1)
        logits = self.frequency_classifier(embedding)
        consistency_logits = self.frequency_consistency_head(embedding)
        return logits, embedding, consistency_logits, [out1, out2, out3, out4]

    def wavelets(self, x):
        low_frequency, high_frequency = self.dwt(x)
        return self.plugdata(x, low_frequency, high_frequency)

    def plugdata(self, x, low_frequency, high_frequency):
        high = high_frequency[0]
        output = x.new_zeros(
            (
                x.size(0),
                x.size(1) * 4,
                low_frequency.size(2),
                low_frequency.size(3),
            )
        )
        output[:, 0:3, :, :] = low_frequency
        output[:, 3:6, :, :] = high[:, 0, :, :, :]
        output[:, 6:9, :, :] = high[:, 1, :, :, :]
        output[:, 9:12, :, :] = high[:, 2, :, :, :]
        return output


def ResNet18(num_classes: int = 10, flag: bool = False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, frequency=flag)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet18(10, True).to(device)
    y, embedding, consistency, features = net(torch.randn(1, 3, 32, 32, device=device))
    print(consistency.shape)
    print(y.size(), embedding.size())
    for feature in features:
        print(feature.shape)
