import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    """
    Input:  (B, 3)  -> [age_norm, sex_onehot(2)]
    Output: (B, feature_dim)
    """
    def __init__(self, in_dim=3, hidden_dim=32, feature_dim=16, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MetadataClassifier(nn.Module):
    def __init__(self, num_classes=5, in_dim=3, hidden_dim=32, feature_dim=16):
        super().__init__()
        self.encoder = MetadataEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        out = self.classifier(feat)
        return out


def metadata_mlp(num_classes=5):
    return MetadataClassifier(num_classes=num_classes)