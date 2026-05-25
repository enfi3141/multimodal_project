import torch
import torch.nn as nn

from .resnet_lstm_ptbxl_raw import RawECGEncoder
from .metadata_mlp import MetadataEncoder


class PTBXLRawReconMetaNet(nn.Module):
    """
    Three-branch multimodal diagnostic model.

    batch dict 입력:
    - batch["raw_1lead"]     : (B, 1, T)
    - batch["recon_12lead"]  : (B, 12, T)
    - batch["metadata"]      : (B, 3)

    output:
    - logits                 : (B, num_classes)
    """

    def __init__(
        self,
        num_classes=5,
        raw_feature_dim=128,
        recon_feature_dim=128,
        meta_feature_dim=16,
        meta_in_dim=3,
        meta_hidden_dim=32,
        fusion_hidden=128,
        lstm_hidden=128,
        lstm_layers=1,
        layers=(2, 2, 2),
        base_channels=32,
        dropout=0.2,
        meta_dropout=0.1,
    ):
        super().__init__()

        # Branch 1: original single-lead ECG
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
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, batch):
        raw_1lead = batch["raw_1lead"]          # (B, 1, T)
        recon_12lead = batch["recon_12lead"]   # (B, 12, T)
        metadata = batch["metadata"]           # (B, 3)

        raw_feat = self.raw_1lead_encoder(raw_1lead)
        recon_feat = self.recon_12lead_encoder(recon_12lead)
        meta_feat = self.meta_encoder(metadata)

        fused = torch.cat([raw_feat, recon_feat, meta_feat], dim=1)

        out = self.classifier(fused)
        return out


def ptbxl_raw_recon_meta_net(num_classes=5):
    return PTBXLRawReconMetaNet(num_classes=num_classes)


if __name__ == "__main__":
    model = ptbxl_raw_recon_meta_net(num_classes=5)

    batch = {
        "raw_1lead": torch.randn(4, 1, 1000),
        "recon_12lead": torch.randn(4, 12, 1000),
        "metadata": torch.randn(4, 3),
    }

    out = model(batch)

    print("raw_1lead:", batch["raw_1lead"].shape)
    print("recon_12lead:", batch["recon_12lead"].shape)
    print("metadata:", batch["metadata"].shape)
    print("output:", out.shape)