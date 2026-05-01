"""
UNet1D Baseline for 12-Lead ECG Reconstruction
================================================
기존 train_1to12_ptbxl.py 에 포함되어 있던 기본 U-Net 모델.
독립 모듈로 분리.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D_Baseline(nn.Module):
    def __init__(self, in_ch=1, out_ch=12, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.down1 = nn.Conv1d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)

        self.enc2 = ConvBlock(base_ch * 2, base_ch * 2)
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1)

        self.enc3 = ConvBlock(base_ch * 4, base_ch * 4)
        self.down3 = nn.Conv1d(base_ch * 4, base_ch * 8, kernel_size=4, stride=2, padding=1)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)

        self.up3 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, kernel_size=4, stride=2, padding=1)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out = nn.Conv1d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)
