import torch
import torch.nn as nn


def conv3x3_1d(in_planes, out_planes, stride=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3_1d(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3_1d(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm1d(planes)

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


class RawECGEncoder(nn.Module):
    """
    Input:  (B, 12, T)
    Output: (B, feature_dim)
    """

    def __init__(
        self,
        in_channels=12,
        feature_dim=128,
        lstm_hidden=128,
        lstm_layers=1,
        block=BasicBlock1D,
        layers=(2, 2, 2),
        base_channels=32,
    ):
        super().__init__()

        self.inplanes = base_channels

        # stem
        self.conv1 = nn.Conv1d(
            in_channels,
            base_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # residual encoder
        self.layer1 = self._make_layer_1d(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer_1d(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer_1d(block, base_channels * 4, layers[2], stride=2)

        encoder_out_dim = base_channels * 4 * block.expansion

        self.lstm = nn.LSTM(
            input_size=encoder_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(lstm_hidden, feature_dim)

    def _make_layer_1d(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 12, T)

        # stem
        x = self.conv1(x)       # (B, C, T/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # (B, C, T/4)

        # residual encoder
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)      # (B, C_out, T')

        # LSTM expects (B, T', C_out)
        x = x.transpose(1, 2)

        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]             # (B, lstm_hidden)
        x = self.fc(x)          # (B, feature_dim)

        return x


class ResNetLSTMPTBXLRaw(nn.Module):
    def __init__(
        self,
        num_classes=5,
        feature_dim=128,
        lstm_hidden=128,
        lstm_layers=1,
        layers=(2, 2, 2),
        base_channels=32,
    ):
        super().__init__()

        self.encoder = RawECGEncoder(
            in_channels=12,
            feature_dim=feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            layers=layers,
            base_channels=base_channels,
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        out = self.classifier(feat)
        return out


def resnet_lstm_ptbxl_raw(
    num_classes=5,
    feature_dim=128,
    lstm_hidden=128,
    lstm_layers=1,
    layers=(2, 2, 2),
    base_channels=32,
):
    return ResNetLSTMPTBXLRaw(
        num_classes=num_classes,
        feature_dim=feature_dim,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        layers=layers,
        base_channels=base_channels,
    )