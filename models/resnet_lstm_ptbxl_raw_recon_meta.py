import torch
import torch.nn as nn

from models.resnet_lstm_ptbxl_raw import RawECGEncoder
from models.metadata_mlp import MetadataEncoder


class ResNetLSTMPTBXLRawReconMeta(nn.Module):
    """
    Three-branch ECG diagnostic model.

    Branch 1:
        Original 1-lead ECG
        input shape: (B, 1, T)

    Branch 2:
        Reconstructed 12-lead ECG
        input shape: (B, 12, T)

    Branch 3:
        Metadata
        input shape: (B, 3)
        example: [age_norm, sex_onehot_0, sex_onehot_1]

    Output:
        logits for PTB-XL superclass multi-label classification
        output shape: (B, num_classes)
    """

    def __init__(
        self,
        num_classes=5,
        meta_in_dim=3,
        raw_feature_dim=128,
        recon_feature_dim=128,
        meta_hidden_dim=32,
        meta_feature_dim=16,
        lstm_hidden=128,
        lstm_layers=1,
        layers=(2, 2, 2),
        base_channels=32,
        dropout=0.3,
        meta_dropout=0.1,
    ):
        super().__init__()

        # Branch 1: original 1-lead ECG
        self.raw_1lead_encoder = RawECGEncoder(
            in_channels=1,
            feature_dim=raw_feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            layers=layers,
            base_channels=base_channels,
        )

        # Branch 2: reconstructed 12-lead ECG
        self.recon_12lead_encoder = RawECGEncoder(
            in_channels=12,
            feature_dim=recon_feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            layers=layers,
            base_channels=base_channels,
        )

        # Branch 3: metadata
        self.meta_encoder = MetadataEncoder(
            in_dim=meta_in_dim,
            hidden_dim=meta_hidden_dim,
            feature_dim=meta_feature_dim,
            dropout=meta_dropout,
        )

        fusion_dim = raw_feature_dim + recon_feature_dim + meta_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, raw_1lead, recon_12lead, meta):
        """
        raw_1lead:
            Tensor of shape (B, 1, T)

        recon_12lead:
            Tensor of shape (B, 12, T)

        meta:
            Tensor of shape (B, 3)
        """

        raw_feat = self.raw_1lead_encoder(raw_1lead)
        recon_feat = self.recon_12lead_encoder(recon_12lead)
        meta_feat = self.meta_encoder(meta)

        fused = torch.cat([raw_feat, recon_feat, meta_feat], dim=1)

        logits = self.classifier(fused)
        return logits


def resnet_lstm_ptbxl_raw_recon_meta(
    num_classes=5,
    meta_in_dim=3,
    raw_feature_dim=128,
    recon_feature_dim=128,
    meta_hidden_dim=32,
    meta_feature_dim=16,
    lstm_hidden=128,
    lstm_layers=1,
    layers=(2, 2, 2),
    base_channels=32,
    dropout=0.3,
    meta_dropout=0.1,
):
    return ResNetLSTMPTBXLRawReconMeta(
        num_classes=num_classes,
        meta_in_dim=meta_in_dim,
        raw_feature_dim=raw_feature_dim,
        recon_feature_dim=recon_feature_dim,
        meta_hidden_dim=meta_hidden_dim,
        meta_feature_dim=meta_feature_dim,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        layers=layers,
        base_channels=base_channels,
        dropout=dropout,
        meta_dropout=meta_dropout,
    )


if __name__ == "__main__":
    model = resnet_lstm_ptbxl_raw_recon_meta(num_classes=5)

    raw_1lead = torch.randn(4, 1, 1000)
    recon_12lead = torch.randn(4, 12, 1000)
    meta = torch.randn(4, 3)

    out = model(raw_1lead, recon_12lead, meta)

    print("raw_1lead:", raw_1lead.shape)
    print("recon_12lead:", recon_12lead.shape)
    print("meta:", meta.shape)
    print("output:", out.shape)