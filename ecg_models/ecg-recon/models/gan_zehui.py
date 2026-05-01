"""
GAN for 12-Lead ECG Reconstruction (Zehui Zhan)
================================================
논문: "Conditional generative adversarial network driven variable-duration
      single-lead to 12-lead electrocardiogram reconstruction"
      (Zehui Zhan, Jiarong Chen, Kangming Li, Wanqing Wu)
공식 GitHub: https://github.com/Zehui-Zhan/12-lead-reconstruction

원본 코드: GAN.py (Generator_gan, Discriminator_gan 클래스) 에서 직접 추출.
원본: in=1, out=11 (입력 lead 제외)
수정: in=1, out=12 (전체 12 leads 출력)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────── 원본 빌딩 블록 (GAN.py 17-35줄) ────────────────────────

def gblock(filnum_input, filnum, kernel=4):
    """Encoder block: Conv1d + BN + LeakyReLU"""
    return nn.Sequential(
        nn.Conv1d(in_channels=filnum_input, out_channels=filnum,
                  kernel_size=kernel, stride=2, padding=1),
        nn.BatchNorm1d(filnum),
        nn.LeakyReLU(0.2),
    )


def upblock(filnum_input, filnum, kernel=4):
    """Decoder block with dropout"""
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels=filnum_input, out_channels=filnum,
                           kernel_size=kernel, stride=2, padding=1),
        nn.BatchNorm1d(filnum),
        nn.Dropout(0.1),
        nn.ReLU(),
    )


def upblock_second(filnum_input, filnum, kernel=4):
    """Decoder block without dropout"""
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels=filnum_input, out_channels=filnum,
                           kernel_size=kernel, stride=2, padding=1),
        nn.BatchNorm1d(filnum),
        nn.ReLU(),
    )


def conv(input_ch, output_ch, kernel):
    return nn.Sequential(
        nn.Conv1d(in_channels=input_ch, out_channels=output_ch,
                  kernel_size=kernel, stride=2, padding=1),
        nn.LeakyReLU(0.2),
    )


def upconv(input_ch, output_ch, kernel):
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels=input_ch, out_channels=output_ch,
                           kernel_size=kernel, stride=2, padding=1),
        nn.LeakyReLU(0.2),
    )


# ──────────────────────── 원본 Generator (GAN.py 37-85줄) ────────────────────────

class Generator_Zehui(nn.Module):
    """
    원본 Generator_gan 구조 그대로 + 출력 12 leads로 변경.
    U-Net 스타일 인코더-디코더 with skip connections.
    """
    def __init__(self, length=1024, in_channels=1, out_channels=12):
        super().__init__()
        self.gblock1 = gblock(64, 128)
        self.gblock2 = gblock(128, 256)
        self.gblock3 = gblock(256, 512)
        self.gblock4 = gblock(512, 512)
        self.gblock5 = gblock(512, 512)
        self.gblock6 = gblock(512, 512)
        self.gblock7 = gblock(512, 1024)
        self.upblock1 = upblock(1024, 512)
        self.upblock2 = upblock(1024, 512)
        self.upblock3 = upblock(1024, 512)
        self.upblock4 = upblock_second(1024, 512)
        self.upblock5 = upblock_second(1024, 256)
        self.upblock6 = upblock_second(512, 128)
        self.upblock7 = upblock_second(256, 64)
        self.conv = conv(in_channels, 64, 4)
        self.upconv = upconv(128, out_channels, 4)

    @staticmethod
    def _match(a, b):
        """Skip connection 길이 맞춤 (비-2^n 길이 지원)"""
        min_len = min(a.size(2), b.size(2))
        return a[:, :, :min_len], b[:, :, :min_len]

    def forward(self, z):
        """
        Args:
            z: (B, 1, T) - 단일 리드 ECG
        Returns:
            (B, 12, T) - 복원된 12-리드 ECG
        """
        T_in = z.size(2)
        x1 = self.conv(z)
        x2 = self.gblock1(x1)
        x3 = self.gblock2(x2)
        x4 = self.gblock3(x3)
        x5 = self.gblock4(x4)
        x6 = self.gblock5(x5)
        x7 = self.gblock6(x6)
        x8 = self.gblock7(x7)

        y1 = self.upblock1(x8)
        y1, x7 = self._match(y1, x7)
        y1 = torch.cat((y1, x7), dim=1)
        y2 = self.upblock2(y1)
        y2, x6 = self._match(y2, x6)
        y2 = torch.cat((y2, x6), dim=1)
        y3 = self.upblock3(y2)
        y3, x5 = self._match(y3, x5)
        y3 = torch.cat((y3, x5), dim=1)
        y4 = self.upblock4(y3)
        y4, x4 = self._match(y4, x4)
        y4 = torch.cat((y4, x4), dim=1)
        y5 = self.upblock5(y4)
        y5, x3 = self._match(y5, x3)
        y5 = torch.cat((y5, x3), dim=1)
        y6 = self.upblock6(y5)
        y6, x2 = self._match(y6, x2)
        y6 = torch.cat((y6, x2), dim=1)
        y7 = self.upblock7(y6)
        y7, x1 = self._match(y7, x1)
        y7 = torch.cat((y7, x1), dim=1)

        output = self.upconv(y7)
        # 길이 복원
        if output.size(2) != T_in:
            output = F.interpolate(output, size=T_in, mode='linear',
                                   align_corners=False)
        return output


# ──────────────────────── 원본 Discriminator (GAN.py 86-103줄) ────────────────────────

class Discriminator_Zehui(nn.Module):
    """원본 Discriminator_gan 구조 그대로."""
    def __init__(self, in_channels=12, length=1024):
        super().__init__()
        self.down1 = gblock(in_channels, 256)
        self.down2 = gblock(256, 128)
        self.down3 = gblock(128, 64)
        self.down4 = gblock(64, 32)
        self.down5 = gblock(32, 32)
        self.down6 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=1,
                      kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x1 = self.down1(z)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        return self.down6(x5)


# ──────────────────────── 원본 Loss 함수 (GAN.py 105-117줄) ────────────────────────

def gen_loss(disc_generated_output, gen_output, target):
    gan_loss = torch.mean(torch.ones_like(disc_generated_output) - disc_generated_output)
    l1_loss = torch.mean(torch.abs(target - gen_output))
    return gan_loss + l1_loss * 2


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = torch.mean(torch.ones_like(disc_real_output) - disc_real_output)
    fake_loss = torch.mean(disc_generated_output)
    return real_loss + fake_loss
