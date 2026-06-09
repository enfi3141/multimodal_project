import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, feature_dim=128, dropout=0.2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h = self.block1(x)
        h = h + self.block2(h)
        return self.out(h)


class MetadataClassifier(nn.Module):
    def __init__(self, num_classes=5, in_dim=6, hidden_dim=256, feature_dim=128, dropout=0.2):
        super().__init__()

        self.encoder = MetadataEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            dropout=dropout,
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)


def metadata_mlp(num_classes=5, in_dim=6):
    return MetadataClassifier(
        num_classes=num_classes,
        in_dim=in_dim,
        hidden_dim=256,
        feature_dim=128,
        dropout=0.2,
    )