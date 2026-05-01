"""
VAE for 12-Lead ECG Reconstruction (Zehui Zhan)
=================================================
논문: "Conditional generative adversarial network driven variable-duration
      single-lead to 12-lead electrocardiogram reconstruction"
      (Zehui Zhan, Jiarong Chen, Kangming Li, Wanqing Wu)
공식 GitHub: https://github.com/Zehui-Zhan/12-lead-reconstruction

원본 코드: VAE_CNN.py (VAE 클래스) 에서 직접 추출.
원본: in=1, out=11 (입력 lead 제외)
수정: in=1, out=12 (전체 12 leads 출력), latent_dim 유지
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_Zehui(nn.Module):
    """
    원본 VAE 구조 그대로 래핑.
    Conv1D 인코더 → μ/σ → reparameterization → Conv1D 디코더
    """
    def __init__(self, hiddens=[16, 32, 64, 128, 256], latent_dim=128,
                 seq_len=1000, out_channels=12):
        super().__init__()

        # ──── Encoder (원본 VAE_CNN.py 25-38줄) ────
        prev_channels = 1
        modules = []
        img_length = seq_len
        for cur_channels in hiddens:
            modules.append(nn.Sequential(
                nn.Conv1d(prev_channels, cur_channels, kernel_size=3,
                          stride=2, padding=1),
                nn.BatchNorm1d(cur_channels),
                nn.ReLU(),
            ))
            prev_channels = cur_channels
            img_length = (img_length + 1) // 2  # ceil division for stride=2
        self.encoder = nn.Sequential(*modules)

        self.mean_linear = nn.Linear(prev_channels * img_length, latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length, latent_dim)
        self.latent_dim = latent_dim

        # ──── Decoder (원본 VAE_CNN.py 47-73줄) ────
        modules = []
        self.decoder_projection = nn.Linear(latent_dim, prev_channels * img_length)
        self.decoder_input_chw = (prev_channels, img_length)

        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(hiddens[i], hiddens[i - 1], kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(hiddens[i - 1]),
                nn.ReLU(),
            ))
        modules.append(nn.Sequential(
            nn.ConvTranspose1d(hiddens[0], hiddens[0], kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(hiddens[0]),
            nn.ReLU(),
            nn.Conv1d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ))
        self.decoder = nn.Sequential(*modules)
        # 수정: 원본 out=11 → 12
        self.conv_out = nn.Conv1d(3, out_channels, kernel_size=3, stride=1, padding=1)
        self._seq_len = seq_len

    def forward(self, x):
        """
        Args:
            x: (B, 1, T) - 단일 리드 ECG
        Returns:
            decoded: (B, 12, T) - 복원된 12-리드 ECG
            mean: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)

        # 💡 [버그 수정] logvar가 너무 커져서 exp(logvar)가 무한대(inf)가 되어 NaN이 뜨는 것을 방지
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)

        # Reparameterization trick (원본 80-82줄)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean

        dec = self.decoder_projection(z)
        dec = torch.reshape(dec, (-1, *self.decoder_input_chw))
        decoded = self.decoder(dec)
        decoded = self.conv_out(decoded)

        # 길이 조정
        if decoded.size(2) != self._seq_len:
            decoded = F.interpolate(decoded, size=self._seq_len, mode='linear',
                                    align_corners=False)
        return decoded, mean, logvar


def vae_loss(y, y_hat, mean, logvar, kl_weight=0.0):
    """원본 loss_fn (VAE_CNN.py 96-100줄)"""
    recons_loss = F.mse_loss(y_hat, y)
    
    # kl_weight가 0일 때는 굳이 계산해서 inf * 0 = NaN 에러를 낼 위험을 차단
    if kl_weight == 0.0:
        return recons_loss * 100.0
        
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0
    )
    return recons_loss * 100.0 + kl_loss * kl_weight
