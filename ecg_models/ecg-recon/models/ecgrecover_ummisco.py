"""
ECGrecover: U-Net Hybrid Autoencoder for 12-Lead ECG Reconstruction
=====================================================================
논문: "ECGrecover: A Deep Learning Approach for
      Electrocardiogram Signal Completion" (UMMISCO, 2024)
공식 GitHub: https://github.com/UMMISCO/ecgrecover

원본 코드: tools/LoadModel.py (Autoencoder_net 클래스) 에서 직접 추출.
구조: 1D + 2D Conv를 병렬로 사용하는 하이브리드 U-Net Autoencoder.
수정: 원본은 (B,1,12,T) → (B,12,T) 형태였으나,
     1→12 리드 복원을 위해 입력 (B,1,T) → 내부적으로 (B,1,12,T) 확장 → (B,12,T) 출력
"""

import torch
import torch.nn as nn
import numpy as np


# ──────────────────────── 원본 코드 (tools/LoadModel.py 7-24줄) ────────────────────────

class Convolution1D_layer(nn.Module):
    """원본 그대로: 각 리드에 독립적으로 1D Conv 적용"""
    def __init__(self, in_f, out_f, n_leads=12):
        super().__init__()
        self.f = out_f
        self.n_leads = n_leads
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_f, out_channels=out_f,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=out_f),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
        )

    def forward(self, x, device):
        b = x.size(0)
        T_out = x.size(-1) // 2
        new_x = torch.zeros(b, self.f, self.n_leads, T_out,
                            dtype=x.dtype, device=device)
        for i in range(self.n_leads):
            new_x[:, :, i, :] = self.conv(x[:, :, i, :])
        return new_x


class Deconvolution1D_layer(nn.Module):
    """원본 그대로: 각 리드에 독립적으로 1D DeConv 적용"""
    def __init__(self, in_f, out_f, n_leads=12):
        super().__init__()
        self.f = out_f
        self.n_leads = n_leads
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_f, out_channels=out_f,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=out_f),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
        )

    def forward(self, x, device):
        b = x.size(0)
        T_out = x.size(-1) * 2
        new_x = torch.zeros(b, self.f, self.n_leads, T_out,
                            dtype=x.dtype, device=device)
        for i in range(self.n_leads):
            new_x[:, :, i, :] = self.deconv(x[:, :, i, :])
        return new_x


class Convolution2D_layer(nn.Module):
    """원본 그대로: 리드 간 관계 포착하는 2D Conv"""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_f, out_channels=out_f,
                      kernel_size=(13, 4), stride=(1, 2), padding=(6, 1)),
            nn.BatchNorm2d(num_features=out_f),
            nn.LeakyReLU(0.02),
        )

    def forward(self, x):
        return self.conv(x)


class Deconvolution2D_layer(nn.Module):
    """원본 그대로"""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_f, out_channels=out_f,
                               kernel_size=(13, 4), stride=(1, 2), padding=(6, 1)),
            nn.BatchNorm2d(num_features=out_f),
            nn.LeakyReLU(0.02),
        )

    def forward(self, x):
        return self.deconv(x)


# ──────────────────────── 원본 Autoencoder (tools/LoadModel.py 80-179줄) ────────────────────────

class Autoencoder_net(nn.Module):
    """원본 ECGrecover Autoencoder 구조 그대로."""
    def __init__(self):
        super().__init__()
        self.first_conv2D = Convolution2D_layer(1, 16)
        self.first_conv1D = Convolution1D_layer(1, 16)

        self.second_conv2D = Convolution2D_layer(16, 32)
        self.second_conv1D = Convolution1D_layer(16, 32)

        self.third_conv2D = Convolution2D_layer(32, 64)
        self.third_conv1D = Convolution1D_layer(32, 64)

        self.fourth_conv2D = Convolution2D_layer(64, 128)
        self.fourth_conv1D = Convolution1D_layer(64, 128)

        self.first_deconv1D = Deconvolution1D_layer(256, 128)
        self.first_deconv2D = Deconvolution2D_layer(256, 128)

        self.second_deconv1D = Deconvolution1D_layer(256, 64)
        self.second_deconv2D = Deconvolution2D_layer(256, 64)

        self.third_deconv1D = Deconvolution1D_layer(128, 32)
        self.third_deconv2D = Deconvolution2D_layer(128, 32)

        self.fourth_deconv1D = Deconvolution1D_layer(64, 1)
        self.fourth_deconv2D = Deconvolution2D_layer(64, 1)

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1,
                               kernel_size=(13, 3), stride=(1, 1), padding=(6, 1)),
            nn.Tanh(),
        )

        self.transition_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256,
                               kernel_size=(13, 3), stride=(1, 1), padding=(6, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
        )

    @staticmethod
    def _match4d(a, b):
        """4D 텐서의 마지막 차원(시간) 길이 맞춤"""
        min_len = min(a.size(-1), b.size(-1))
        return a[..., :min_len], b[..., :min_len]

    def forward(self, x, device):
        """원본 forward: (B, 1, 12, T) → (B, 12, T)"""
        conv2D_1 = self.first_conv2D(x)
        conv1D_1 = self.first_conv1D(x, device)
        conv1D_1, conv2D_1_m = self._match4d(conv1D_1, conv2D_1)
        conv_1 = torch.cat((conv1D_1, conv2D_1_m), axis=1)

        conv2D_2 = self.second_conv2D(conv2D_1)
        conv1D_2 = self.second_conv1D(conv1D_1, device)
        conv1D_2, conv2D_2_m = self._match4d(conv1D_2, conv2D_2)
        conv_2 = torch.cat((conv1D_2, conv2D_2_m), axis=1)

        conv2D_3 = self.third_conv2D(conv2D_2)
        conv1D_3 = self.third_conv1D(conv1D_2, device)
        conv1D_3, conv2D_3_m = self._match4d(conv1D_3, conv2D_3)
        conv_3 = torch.cat((conv1D_3, conv2D_3_m), axis=1)

        conv2D_4 = self.fourth_conv2D(conv2D_3)
        conv1D_4 = self.fourth_conv1D(conv1D_3, device)
        conv1D_4, conv2D_4_m = self._match4d(conv1D_4, conv2D_4)
        conv_4 = torch.cat((conv1D_4, conv2D_4_m), axis=1)

        transition = self.transition_block(conv_4)

        deconv2D_1 = self.first_deconv2D(conv_4)
        deconv2D_1, conv_3 = self._match4d(deconv2D_1, conv_3)
        deconv_1 = torch.cat((deconv2D_1, conv_3), axis=1)

        deconv2D_2 = self.second_deconv2D(deconv_1)
        deconv2D_2, conv_2 = self._match4d(deconv2D_2, conv_2)
        deconv_2 = torch.cat((deconv2D_2, conv_2), axis=1)

        deconv2D_3 = self.third_deconv2D(deconv_2)
        deconv2D_3, conv_1 = self._match4d(deconv2D_3, conv_1)
        deconv_3 = torch.cat((deconv2D_3, conv_1), axis=1)

        deconv2D_4 = self.fourth_deconv2D(deconv_3)

        out = self.final_conv(deconv2D_4)
        out = torch.squeeze(out, 1)  # (B, 12, T)
        return out


# ──────────────────────── 1→12 래핑 모델 ────────────────────────

class ECGrecover_UMMISCO(nn.Module):
    """
    원본 ECGrecover의 Autoencoder_net을 래핑.
    입력: (B, 1, T) → 단일 리드를 12채널로 확장 → (B, 1, 12, T) → Autoencoder → (B, 12, T)
    """
    def __init__(self):
        super().__init__()
        self.autoencoder = Autoencoder_net()

    def forward(self, x):
        """
        Args:
            x: (B, 1, T) - 단일 리드 ECG
        Returns:
            (B, 12, T) - 복원된 12-리드 ECG
        """
        device = x.device
        B, _, T = x.shape
        # 단일 리드를 12채널로 복제하여 입력 형태 맞춤
        x_expanded = x.unsqueeze(2).expand(B, 1, 12, T)  # (B, 1, 12, T)
        out = self.autoencoder(x_expanded, device)
        # 길이 조정 (stride 때문에 달라질 수 있음)
        if out.size(2) != T:
            out = torch.nn.functional.interpolate(out, size=T, mode='linear',
                                                   align_corners=False)
        return out
