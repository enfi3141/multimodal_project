"""
U-Net 1D for 12-Lead ECG Reconstruction
========================================
논문: "12 Leads ECG Signal Reconstruction from Single Lead" (Gavin Jang, 2023)
공식 GitHub: https://github.com/hawkiyc/12_Leads_ECG_Reconstruction_wtih_U_Net
참고: Chen et al., "Learning to See in the Dark" (2018) 에서 영감.

원본 코드에서 직접 추출하여 1→12 리드 인터페이스로 래핑.
원본: Lead II(1ch) → 나머지 11 leads 복원
수정: 임의 1 lead(1ch) → 전체 12 leads 복원
"""

import torch
import torch.nn as nn


# ──────────────────────── 원본 코드 (hawkiyc/train.py 220-296줄) ────────────────────────

class DoubleConv(nn.Module):
    """Conv -> BN -> LReLU -> Conv -> BN -> LReLU (원본 그대로)"""
    def __init__(self, in_ch, out_ch, droprate=0.05):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 19, padding='same'),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(droprate),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, 19, padding='same'),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(droprate),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.f(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, droprate=0.05):
        super().__init__()
        self.f = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_ch, out_ch, droprate),
        )

    def forward(self, x):
        return self.f(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, droprate=0.05):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, droprate)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        # 길이가 맞지 않을 경우 패딩
        diff = x2.size(2) - x1.size(2)
        if diff > 0:
            x1 = nn.functional.pad(x1, [0, diff])
        elif diff < 0:
            x2 = nn.functional.pad(x2, [0, -diff])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.f = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.f(x)


# ──────────────────────── 래핑 모델 ────────────────────────

class UNet_Hawkiyc(nn.Module):
    """
    원본 Unet 구조를 그대로 유지하되, 출력을 12 leads로 변경.
    원본: in=1, out=11 (입력 lead 제외 나머지)
    수정: in=1, out=12 (전체 12 leads 출력)
    """
    def __init__(self, in_ch=1, out_ch=12, droprate=0.05):
        super().__init__()
        self.inc = DoubleConv(in_ch, 32, droprate)
        self.d1 = Down(32, 64, droprate)
        self.d2 = Down(64, 128, droprate)
        self.d3 = Down(128, 256, droprate)
        self.d4 = Down(256, 512, droprate)

        self.u1 = Up(512, 256, droprate)
        self.u2 = Up(256, 128, droprate)
        self.u3 = Up(128, 64, droprate)
        self.u4 = Up(64, 32, droprate)
        self.outc = OutConv(32, out_ch)

    def forward(self, x):
        """
        Args:
            x: (B, 1, T) - 단일 리드 ECG
        Returns:
            (B, 12, T) - 복원된 12-리드 ECG
        """
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)
