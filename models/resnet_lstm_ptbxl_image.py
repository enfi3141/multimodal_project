from __future__ import absolute_import

import math
import torch
import torch.nn as nn

__all__ = ["resnet_lstm_ptbxl_image"]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ImageEncoder(nn.Module):
    """
    Input:  (B, T, 3, H, W)
    Output: (B, feature_dim)
    """

    def __init__(
        self,
        depth=20,
        feature_dim=128,
        lstm_hidden=128,
        lstm_layers=2,
        bidirectional=True,
    ):
        super().__init__()

        assert (depth - 2) % 6 == 0, "depth should be 6n+2"
        n = (depth - 2) // 6
        block = BasicBlock

        self.block = block
        self.inplanes = 16
        self.bidirectional = bidirectional
        self.lstm_hidden = lstm_hidden

        # stem
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # residual encoder
        self.layer1 = self._make_layer(block, 16, n, stride=1)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        cnn_out_dim = 64 * block.expansion

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.fc = nn.Linear(lstm_out_dim, feature_dim)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / k))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # x: (B, T, 3, H, W)
        batch_size, time_steps, c, h, w = x.shape
        x = x.view(batch_size * time_steps, c, h, w)

        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # residual CNN encoder
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # spatial pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)              # (B*T, cnn_out_dim)
        x = x.view(batch_size, time_steps, -1) # (B, T, cnn_out_dim)

        # temporal modeling
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            x = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2*lstm_hidden)
        else:
            x = h_n[-1]                               # (B, lstm_hidden)

        x = self.fc(x)                                # (B, feature_dim)
        return x


class ResNetLSTMPTBXLImage(nn.Module):
    def __init__(
        self,
        depth=20,
        num_classes=5,
        feature_dim=128,
        lstm_hidden=128,
        lstm_layers=2,
        bidirectional=True,
    ):
        super().__init__()

        self.encoder = ImageEncoder(
            depth=depth,
            feature_dim=feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        out = self.classifier(feat)
        return out


def resnet_lstm_ptbxl_image(**kwargs):
    return ResNetLSTMPTBXLImage(**kwargs)