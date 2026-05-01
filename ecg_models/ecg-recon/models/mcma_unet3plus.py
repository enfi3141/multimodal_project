"""
MCMA: Multi-Channel Masked Autoencoder (UNet3+ backbone)
=========================================================
논문: "Multi-channel masked autoencoder for 12-lead ECG reconstruction
      from arbitrary single-lead" (Chen et al., npj Cardiovascular Health 2024)
공식 GitHub: https://github.com/CHENJIAR3/MCMA

원본 코드: model.py (downblock, upblock, unet3plus_block, modelx) 에서 직접 추출.
원본: TensorFlow/Keras → PyTorch로 변환 (구조/파라미터 동일 유지)
원본: in=(1024,12), out=(1024,12) — Masked Autoencoder (마스킹된 리드 복원)
수정: in=(B,1,T), out=(B,12,T) — 1리드 → 12리드 복원

⚠️ 원본은 TensorFlow 기반이므로, PyTorch로 충실하게 변환.
  원본에서 사용된 InstanceNorm, GELU, LayerNorm은 모두 PyTorch equivalent로 교체.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    """
    원본 downblock (model.py 4-25줄) PyTorch 변환.
    구조: 두 개의 parallel path → Instance Norm으로 합침
    Path 1: Conv1D + GELU
    Path 2: Conv1D + LayerNorm + GELU → Conv1D + GELU
    출력: InstanceNorm(path1 + path2_out)
    """
    def __init__(self, in_ch, out_ch, kernel_size=13, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2

        # Path 1 (원본 5-8줄)
        self.path1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding),
            nn.GELU(),
        )
        # Path 2 (원본 11-22줄)
        self.path2_conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.path2_ln = nn.LayerNorm(out_ch)  # applied after permute
        self.path2_gelu = nn.GELU()
        self.path2_conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, 1, padding),
            nn.GELU(),
        )
        # Instance Norm (원본 23줄: tfa.layers.InstanceNormalization)
        self.inorm = nn.InstanceNorm1d(out_ch, eps=1e-9)

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2_conv(x)
        # LayerNorm: (B, C, T) → (B, T, C) → LN → (B, C, T)
        x2 = x2.permute(0, 2, 1)
        x2 = self.path2_ln(x2)
        x2 = x2.permute(0, 2, 1)
        x2 = self.path2_gelu(x2)
        x2 = self.path2_conv2(x2)
        return self.inorm(x1 + x2)


class UpBlock(nn.Module):
    """
    원본 upblock (model.py 27-48줄) PyTorch 변환.
    DownBlock과 동일 구조, Conv → ConvTranspose 교체.
    """
    def __init__(self, in_ch, out_ch, kernel_size=13, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        out_pad = stride - 1 if stride > 1 else 0

        self.path1 = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride, padding,
                               output_padding=out_pad),
            nn.GELU(),
        )
        self.path2_deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size,
                                                stride, padding,
                                                output_padding=out_pad)
        self.path2_ln = nn.LayerNorm(out_ch)
        self.path2_gelu = nn.GELU()
        self.path2_conv = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, 1, padding),
            nn.GELU(),
        )
        self.inorm = nn.InstanceNorm1d(out_ch, eps=1e-9)

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2_deconv(x)
        x2 = x2.permute(0, 2, 1)
        x2 = self.path2_ln(x2)
        x2 = x2.permute(0, 2, 1)
        x2 = self.path2_gelu(x2)
        x2 = self.path2_conv(x2)
        return self.inorm(x1 + x2)


class MCMA_UNet3Plus(nn.Module):
    """
    원본 unet3plus_block + modelx (model.py 50-94줄) PyTorch 변환.
    구조: 6단계 인코더 + 5단계 디코더 (UNet3+ 스타일 skip connection)
    
    1리드 → 12리드 복원을 위해:
    - 입력 채널: 1 (원본은 12)
    - 출력 채널: 12
    """
    def __init__(self, input_ch=1, output_ch=12, pool_size=2, kernel_size=13):
        super().__init__()
        filters = [16, 32, 64, 128, 256, 512]

        # Encoder (원본 51-56줄)
        self.e1 = DownBlock(input_ch, filters[0], kernel_size, stride=1)
        self.e2 = DownBlock(filters[0], filters[1], kernel_size, stride=pool_size)
        self.e3 = DownBlock(filters[1], filters[2], kernel_size, stride=pool_size)
        self.e4 = DownBlock(filters[2], filters[3], kernel_size, stride=pool_size)
        self.e5 = DownBlock(filters[3], filters[4], kernel_size, stride=pool_size)
        self.e6 = DownBlock(filters[4], filters[5], kernel_size, stride=pool_size)

        # Decoder (원본 58-78줄)
        self.d5_up = UpBlock(filters[5], filters[4], kernel_size, stride=pool_size)
        self.d5_skip = DownBlock(filters[4], filters[4], kernel_size)

        self.d4_up = UpBlock(filters[4], filters[3], kernel_size, stride=pool_size)
        self.d4_skip = DownBlock(filters[3], filters[3], kernel_size)

        self.d3_up = UpBlock(filters[3], filters[2], kernel_size, stride=pool_size)
        self.d3_skip = DownBlock(filters[2], filters[2], kernel_size)

        self.d2_up = UpBlock(filters[2], filters[1], kernel_size, stride=pool_size)
        self.d2_skip = DownBlock(filters[1], filters[1], kernel_size)

        self.d1_up = UpBlock(filters[1], filters[0], kernel_size, stride=pool_size)
        self.d1_skip = DownBlock(filters[0], filters[0], kernel_size)

        # Output
        self.d0 = DownBlock(filters[0], output_ch, kernel_size)

    def forward(self, x):
        """
        Args:
            x: (B, 1, T) - 단일 리드 ECG
        Returns:
            (B, 12, T) - 복원된 12-리드 ECG
        """
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        # Decoder with skip connections (원본 58-76줄)
        d5_e6 = self.d5_up(e6)
        d5_e5 = self.d5_skip(e5)
        # 길이 맞추기
        min_len = min(d5_e6.size(2), d5_e5.size(2))
        d5 = d5_e5[:, :, :min_len] + d5_e6[:, :, :min_len]

        d4_d5 = self.d4_up(d5)
        d4_e4 = self.d4_skip(e4)
        min_len = min(d4_d5.size(2), d4_e4.size(2))
        d4 = d4_d5[:, :, :min_len] + d4_e4[:, :, :min_len]

        d3_d4 = self.d3_up(d4)
        d3_e3 = self.d3_skip(e3)
        min_len = min(d3_d4.size(2), d3_e3.size(2))
        d3 = d3_d4[:, :, :min_len] + d3_e3[:, :, :min_len]

        d2_d3 = self.d2_up(d3)
        d2_e2 = self.d2_skip(e2)
        min_len = min(d2_d3.size(2), d2_e2.size(2))
        d2 = d2_d3[:, :, :min_len] + d2_e2[:, :, :min_len]

        d1_d2 = self.d1_up(d2)
        d1_e1 = self.d1_skip(e1)
        min_len = min(d1_d2.size(2), d1_e1.size(2))
        d1 = d1_d2[:, :, :min_len] + d1_e1[:, :, :min_len]

        # 출력 & 길이 복원
        out = self.d0(d1)
        T_in = x.size(2)
        if out.size(2) != T_in:
            out = F.interpolate(out, size=T_in, mode='linear', align_corners=False)
        return out
