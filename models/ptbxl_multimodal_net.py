import torch
import torch.nn as nn

from .resnet_lstm_ptbxl_raw import ResNetLSTMPTBXLRaw
from .resnet_lstm_ptbxl_image import ResNetLSTMPTBXLImage
from .metadata_mlp import MetadataEncoder


class PTBXLMultimodalNet(nn.Module):
    """
    batch dict 입력:
    - batch["ecg_raw"]   : (B, 12, T)
    - batch["ecg_img"]   : (B, T_img, 3, H, W)
    - batch["metadata"]  : (B, 3)
    """
    def __init__(
        self,
        num_classes=5,
        raw_feature_dim=128,
        image_feature_dim=128,
        meta_feature_dim=16,
        fusion_hidden=128,
    ):
        super().__init__()

        self.raw_model = ResNetLSTMPTBXLRaw(num_classes=num_classes, feature_dim=raw_feature_dim)
        self.image_model = ResNetLSTMPTBXLImage(num_classes=num_classes, feature_dim=image_feature_dim)
        self.meta_encoder = MetadataEncoder(feature_dim=meta_feature_dim)

        fusion_dim = raw_feature_dim + image_feature_dim + meta_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, batch):
        raw_x = batch["ecg_raw"]       # (B, 12, T)
        img_x = batch["ecg_img"]       # (B, T_img, 3, H, W)
        meta_x = batch["metadata"]     # (B, 3)

        raw_feat = self.raw_model.encoder(raw_x)
        img_feat = self.image_model.encoder(img_x)
        meta_feat = self.meta_encoder(meta_x)

        fused = torch.cat([raw_feat, img_feat, meta_feat], dim=1)
        out = self.classifier(fused)
        return out


def ptbxl_multimodal_net(num_classes=5):
    return PTBXLMultimodalNet(num_classes=num_classes)