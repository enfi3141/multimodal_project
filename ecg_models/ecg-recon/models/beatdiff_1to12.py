# -*- coding: utf-8 -*-
"""
BeatDiff 1→12 Lead ECG Reconstruction (PyTorch Port)
====================================================
원본: https://github.com/LisaBedin/BeatDiff (JAX/Flax)
논문: "Leveraging an ECG Beat Diffusion Model for Morphological
       Reconstruction from Indirect Signals" (NeurIPS 2024)

이 파일은 BeatDiff의 핵심 구성요소를 PyTorch로 포팅한 것입니다:
    1. DhariwalUnet (EDM-style 1D U-Net) - 원본의 unet_parts.py
    2. DenoiserNet (EDM preconditioning wrapper) - 원본의 unet_parts.py
    3. VE (Variance Exploding) noise schedule - 원본의 variance_exploding_utils.py
    4. Heun 2nd-order ODE sampler - 원본의 variance_exploding_utils.py
    5. Conditional inpainting for 1-to-12 lead - 원본의 ecg_inpainting/

    변형사항 (원본 대비):
    [변형 1] JAX/Flax -> PyTorch 전환 (nn.Module, torch.Tensor 사용)
    [변형 2] 원본은 beat-level (단일 심박) 처리 -> 여기서는 full-length signal 처리
    [변형 3] 원본의 unconditional training + EM inpainting ->
             conditional training (1-lead condition)
"""
from __future__ import annotations

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# EDM Preconditioning Functions (원본: variance_exploding_utils.py)
# ──────────────────────────────────────────────────────────────────────────────
def skip_scaling(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """c_skip(σ): 원본 유지"""
    return (sigma_data ** 2) / (sigma_data ** 2 + sigma ** 2)


def output_scaling(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """c_out(σ): 원본 유지"""
    return sigma * sigma_data / ((sigma_data ** 2 + sigma ** 2) ** 0.5)


def input_scaling(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """c_in(σ): 원본 유지"""
    return 1.0 / ((sigma_data ** 2 + sigma ** 2) ** 0.5)


def noise_conditioning(sigma: torch.Tensor) -> torch.Tensor:
    """c_noise(σ): 원본 유지 - log(σ)/4"""
    return torch.log(sigma) / 4.0


# ──────────────────────────────────────────────────────────────────────────────
# Positional Embedding (원본: unet_parts.py PositionalEmbedding)
# ──────────────────────────────────────────────────────────────────────────────
class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding. 원본과 동일한 수식."""
    def __init__(self, num_channels: int, max_positions: int = 10_000):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.num_channels // 2
        freqs = torch.arange(half, device=x.device, dtype=x.dtype)
        freqs = freqs / (half - 1)
        freqs = (1.0 / self.max_positions) ** freqs
        x_outer = x.unsqueeze(-1) * freqs.unsqueeze(0)  # [*, half]
        return torch.cat([torch.cos(x_outer), torch.sin(x_outer)], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# UNetBlock (원본: unet_parts.py UNetBlock)
# [변형 5] JAX의 [B,L,C] → PyTorch [B,C,L] 변환
# ──────────────────────────────────────────────────────────────────────────────
class UNetBlock(nn.Module):
    """원본 UNetBlock의 PyTorch 포팅.
    GroupNorm + Conv1d + SiLU + optional Self-Attention."""

    def __init__(self, in_ch: int, out_ch: int, emb_ch: int,
                 num_heads: int = 0, dropout_rate: float = 0.1,
                 down: bool = False, up: bool = False, skip_scale: float = 1.0):
        super().__init__()
        self.skip_scale = skip_scale

        # 메인 경로
        if not up:
            stride = 2 if down else 1
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
            self.skip_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.conv1 = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            self.skip_conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

        g1 = min(32, in_ch)
        while in_ch % g1 != 0 and g1 > 1:
            g1 -= 1
        g2 = min(32, out_ch)
        while out_ch % g2 != 0 and g2 > 1:
            g2 -= 1

        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.time_proj = nn.Linear(emb_ch, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)

        # Self-attention (원본과 동일: num_heads > 0일 때만)
        self.has_attn = num_heads > 0
        if self.has_attn:
            g_attn = min(32, out_ch)
            while out_ch % g_attn != 0 and g_attn > 1:
                g_attn -= 1
            self.attn_norm = nn.GroupNorm(g_attn, out_ch)
            self.attn = nn.MultiheadAttention(out_ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L], emb: [B, L_emb, emb_ch] or [B, emb_ch]"""
        skip = self.skip_conv(x)

        h = self.conv1(x)
        # time embedding 투영: 원본과 동일하게 emb가 [B, L, C]일 때 position-wise 추가
        t_emb = self.time_proj(emb)  # [B, L', out_ch] or [B, out_ch]
        if t_emb.ndim == 3:
            # Downsample embedding to match h's length
            if t_emb.size(1) != h.size(2):
                t_emb = F.adaptive_avg_pool1d(t_emb.permute(0, 2, 1), h.size(2)).permute(0, 2, 1)
            h = h + t_emb.permute(0, 2, 1)  # [B, out_ch, L]
        else:
            h = h + t_emb.unsqueeze(-1)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(self.dropout(h))

        h = (h + skip) * self.skip_scale

        if self.has_attn:
            res = h
            h_perm = self.attn_norm(h).permute(0, 2, 1)  # [B, L, C]
            h_perm, _ = self.attn(h_perm, h_perm, h_perm)
            h = (h_perm.permute(0, 2, 1) + res) * self.skip_scale

        return h


# ──────────────────────────────────────────────────────────────────────────────
# DhariwalUnet (원본: unet_parts.py DhariwalUnet)
# ──────────────────────────────────────────────────────────────────────────────
class DhariwalUnet1D(nn.Module):
    """BeatDiff 원본의 DhariwalUnet을 PyTorch로 포팅.
    원본 구조를 최대한 유지하되, [변형 5]로 인해 channel axis가 다릅니다."""

    def __init__(
        self,
        in_ch: int = 12,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 3, 4),
        channel_mult_emb: int = 4,
        num_blocks: int = 3,
        attn_resolutions: tuple = (32, 16, 8),
        dropout_rate: float = 0.10,
        embed_position_in_signal: bool = True,
        conditional: bool = True,
        cond_dim: int = 4,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_blocks = num_blocks
        self.attn_resolutions = attn_resolutions
        self.embed_position_in_signal = embed_position_in_signal
        self.conditional = conditional

        emb_ch = model_channels * channel_mult_emb

        # Noise embedding (원본: PositionalEmbedding + 2-layer MLP)
        self.noise_emb = PositionalEmbedding(model_channels)
        self.noise_mlp = nn.Sequential(
            nn.Linear(model_channels, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )

        # Class conditioning (원본과 동일)
        if conditional:
            self.class_mlp = nn.Sequential(
                nn.Linear(cond_dim, 2 * emb_ch),
                nn.SiLU(),
                nn.Linear(2 * emb_ch, emb_ch),
            )

        # Position embedding in signal (원본과 동일)
        if embed_position_in_signal:
            self.pos_emb = PositionalEmbedding(emb_ch)

        # Encoder
        self.enc_in = nn.Conv1d(in_ch, model_channels * channel_mult[0], kernel_size=3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        for level, mult in enumerate(channel_mult):
            ch = model_channels * mult
            # Downsample (level > 0)
            if level > 0:
                res = 1000 >> level  # 대략적인 해상도
                self.downsample_blocks.append(
                    UNetBlock(
                        in_ch=model_channels * channel_mult[level - 1],
                        out_ch=ch, emb_ch=emb_ch,
                        num_heads=ch // 64,
                        down=True, dropout_rate=dropout_rate,
                    )
                )
            else:
                self.downsample_blocks.append(nn.Identity())

            # ResBlocks at this level
            level_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                res = max(8, 1000 >> level)
                level_blocks.append(
                    UNetBlock(
                        in_ch=ch, out_ch=ch, emb_ch=emb_ch,
                        num_heads=ch // 64 if res in attn_resolutions else 0,
                        dropout_rate=dropout_rate,
                    )
                )
            self.encoder_blocks.append(level_blocks)

        # Bottleneck
        bot_ch = model_channels * channel_mult[-1]
        self.bottleneck = nn.ModuleList([
            UNetBlock(bot_ch, bot_ch, emb_ch, num_heads=bot_ch // 64, dropout_rate=dropout_rate),
            UNetBlock(bot_ch, bot_ch, emb_ch, num_heads=bot_ch // 64, dropout_rate=dropout_rate),
        ])

        # Decoder
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            ch = model_channels * mult
            # Upsample (level < max)
            if level < len(channel_mult) - 1:
                self.upsample_blocks.append(
                    UNetBlock(
                        in_ch=model_channels * channel_mult[level + 1],
                        out_ch=ch, emb_ch=emb_ch,
                        num_heads=ch // 64,
                        up=True, dropout_rate=dropout_rate,
                    )
                )
            else:
                self.upsample_blocks.append(nn.Identity())

            # ResBlocks + skip connections
            level_blocks = nn.ModuleList()
            for blk_idx in range(num_blocks + 1):
                # skip connection doubles channels
                block_in = ch * 2 if blk_idx < num_blocks + 1 else ch
                res = max(8, 1000 >> level)
                level_blocks.append(
                    UNetBlock(
                        in_ch=block_in, out_ch=ch, emb_ch=emb_ch,
                        num_heads=ch // 64 if res in attn_resolutions else 0,
                        dropout_rate=dropout_rate,
                    )
                )
            self.decoder_blocks.append(level_blocks)

        # Output
        out_g = min(32, model_channels)
        while model_channels % out_g != 0 and out_g > 1:
            out_g -= 1
        self.out_norm = nn.GroupNorm(out_g, model_channels)
        self.out_conv = nn.Conv1d(model_channels, in_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, noise_cond: torch.Tensor,
                class_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, C, L] 입력 신호
        noise_cond: [B] 노이즈 조건 (log(σ)/4 변환 후)
        class_features: [B, cond_dim] 리드 조건 (1-lead)
        Returns: [B, C, L]
        """
        B, C, L = x.shape

        # Noise embedding
        emb = self.noise_emb(noise_cond)       # [B, model_ch]
        emb = self.noise_mlp(emb)              # [B, emb_ch]

        # Class conditioning
        if self.conditional and class_features is not None:
            class_emb = self.class_mlp(class_features)  # [B, emb_ch]
            emb = emb + class_emb

        # Position embedding in signal dimension
        if self.embed_position_in_signal:
            pos = torch.linspace(0, 1, L, device=x.device, dtype=x.dtype)
            pos_e = self.pos_emb(pos)   # [L, emb_ch]
            emb = emb.unsqueeze(1) + pos_e.unsqueeze(0)  # [B, L, emb_ch]
            emb = F.silu(emb)

        # Encoder
        h = self.enc_in(x)  # [B, ch, L]

        skips = [h]
        for level, (down, blocks) in enumerate(zip(self.downsample_blocks, self.encoder_blocks)):
            if level > 0:
                h = down(h, emb)
            skips.append(h)
            for block in blocks:
                h = block(h, emb)
                skips.append(h)

        # Bottleneck
        for bot_block in self.bottleneck:
            h = bot_block(h, emb)

        # Decoder
        for up, blocks in zip(self.upsample_blocks, self.decoder_blocks):
            if not isinstance(up, nn.Identity):
                h = up(h, emb)
            for block in blocks:
                s = skips.pop()
                # Handle size mismatch from up/downsampling
                if h.size(2) != s.size(2):
                    diff = s.size(2) - h.size(2)
                    if diff > 0:
                        s = s[:, :, :h.size(2)]
                    else:
                        h = h[:, :, :s.size(2)]
                h = torch.cat([h, s], dim=1)
                h = block(h, emb)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# ──────────────────────────────────────────────────────────────────────────────
# BeatDiff1to12: EDM-style conditional diffusion (메인 모델)
# ──────────────────────────────────────────────────────────────────────────────
class BeatDiff1to12(nn.Module):
    """
    BeatDiff의 핵심 Conditional Diffusion 모델 (PyTorch 포팅).

    원본 vs 변형:
      - 원본: unconditional beat-level diffusion + EM-based inpainting으로 1→12
      - 여기: [변형 3] conditional training (1-lead 직접 concat) + VE diffusion으로 1→12
        이유: EM inpainting은 추론 시 매우 느림 (100 particles × 20 steps × EM iterations)
              conditional training은 단일 forward pass로 12-lead 생성 가능

    EDM preconditioning (원본과 동일):
      D(x; σ) = c_skip(σ) * x + c_out(σ) * F(c_in(σ) * x; c_noise(σ))
    """

    def __init__(
        self,
        cond_ch: int = 1,
        out_ch: int = 12,
        base_ch: int = 128,
        channel_mult: tuple = (1, 2, 3, 4),
        num_blocks: int = 2,
        attn_resolutions: tuple = (32, 16, 8),
        dropout_rate: float = 0.10,
        # VE diffusion params (원본 config.yaml의 diffusion 섹션과 동일)
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        rho: float = 7.0,
        # Sampling
        sample_steps: int = 25,
        val_sample_steps: int = 20,
        # Reconstruction loss weight (추가: x0 예측에 대한 보조 loss)
        recon_weight: float = 0.1,
        clip_x0: float = 4.0,
    ):
        super().__init__()
        self.cond_ch = cond_ch
        self.out_ch = out_ch
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p_mean = p_mean
        self.p_std = p_std
        self.rho = rho
        self.sample_steps = sample_steps
        self.val_sample_steps = val_sample_steps
        self.recon_weight = recon_weight
        self.clip_x0 = clip_x0

        # Denoiser UNet: 입력은 noisy 12-lead + condition 1-lead
        self.denoiser = DhariwalUnet1D(
            in_ch=out_ch + cond_ch,
            model_channels=base_ch,
            channel_mult=channel_mult,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            dropout_rate=dropout_rate,
            embed_position_in_signal=True,
            conditional=False,  # class conditioning은 1-lead concat으로 대체
            cond_dim=4,
        )

    def _denoise(self, x_noisy: torch.Tensor, sigma: torch.Tensor,
                 x_cond: torch.Tensor) -> torch.Tensor:
        """EDM-style denoising with preconditioning (원본 DenoiserNet과 동일 구조).

        D(x; σ) = c_skip(σ) * x + c_out(σ) * F_θ(c_in(σ) * x; c_noise(σ))
        """
        c_skip = skip_scaling(sigma, self.sigma_data)      # [B]
        c_out = output_scaling(sigma, self.sigma_data)      # [B]
        c_in = input_scaling(sigma, self.sigma_data)        # [B]
        c_noise = noise_conditioning(sigma)                 # [B]

        # Preconditioned input
        scaled_x = c_in.view(-1, 1, 1) * x_noisy           # [B, 12, L]

        # Concat condition
        net_input = torch.cat([scaled_x, x_cond], dim=1)   # [B, 13, L]

        # F_θ output
        F_out = self.denoiser(net_input, c_noise)           # [B, 13, L]
        F_out = F_out[:, :self.out_ch]                      # [B, 12, L]

        # EDM preconditioning
        denoised = c_skip.view(-1, 1, 1) * x_noisy + c_out.view(-1, 1, 1) * F_out
        return denoised

    def loss_and_pred(self, x_cond: torch.Tensor, y_true: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """VE Training loss (원본 variance_exploding_utils.py make_loss_fn 과 동일).

        원본:
          σ ~ exp(N(p_mean, p_std²)), clipped to [σ_min, σ_max]
          n ~ N(0, I)
          x_noisy = x + σ * n
          loss = w(σ) * ||D(x_noisy; σ) - x||²
          where w(σ) = (σ² + σ_data²) / (σ * σ_data)²
        """
        B = y_true.shape[0]

        # Sample noise levels (원본과 동일한 log-normal 분포)
        log_sigma = torch.randn(B, device=y_true.device) * self.p_std + self.p_mean
        sigma = log_sigma.exp().clamp(self.sigma_min, self.sigma_max)

        # Add noise (원본과 동일: VE 방식 = x + σ*n)
        noise = torch.randn_like(y_true)
        x_noisy = y_true + sigma.view(-1, 1, 1) * noise

        # Denoise
        denoised = self._denoise(x_noisy, sigma, x_cond)

        # Weighted MSE loss (원본과 동일한 weighting)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        per_sample_loss = weight * ((denoised - y_true) ** 2).mean(dim=(1, 2))
        loss_denoise = per_sample_loss.mean()

        # 보조 reconstruction loss (클립핑된 x0 예측에 대한 L1)
        denoised_clipped = denoised.clamp(-self.clip_x0, self.clip_x0)
        loss_recon = F.l1_loss(denoised_clipped, y_true)

        total = loss_denoise + self.recon_weight * loss_recon

        terms = {
            "denoise": loss_denoise.detach(),
            "recon": loss_recon.detach(),
            "total": total.detach(),
        }
        return total, denoised_clipped.detach(), terms

    @torch.no_grad()
    def sample(self, x_cond: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Heun 2nd-order ODE sampler (원본 heun_sampler와 동일한 알고리즘).

        원본:
          t_i = (σ_max^(1/ρ) + i/(N-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
          d_i = (σ'(t)/σ(t) + s'(t)/s(t)) * x - σ'(t)*s(t)/σ(t) * D(x; σ(t))
          x_{i+1} = x_i + Δt * d_i  (Euler)
          d_{i+1} = ... (2nd-order correction)
          x_{i+1} = x_i + Δt * (d_i + d_{i+1}) / 2  (Heun)

        여기서는 s(t)=1 (VE), σ(t)=t 로 단순화:
          d = (1/t) * x - (1/t) * D(x; t) = (x - D(x;t)) / t
        """
        B, _, L = x_cond.shape
        steps = int(steps or self.sample_steps)
        steps = max(2, steps)

        # Time schedule (원본과 동일: σ_max → σ_min)
        step_indices = torch.arange(steps, device=x_cond.device, dtype=x_cond.dtype)
        t_steps = (
            self.sigma_max ** (1.0 / self.rho)
            + step_indices / (steps - 1) * (
                self.sigma_min ** (1.0 / self.rho) - self.sigma_max ** (1.0 / self.rho)
            )
        ) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros(1, device=x_cond.device)])

        # Initial samples from N(0, σ_max²)
        x = torch.randn(B, self.out_ch, L, device=x_cond.device) * self.sigma_max

        for i in range(steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            dt = t_next - t_cur

            sigma_vec = t_cur.expand(B)

            # Denoiser prediction
            d_cur = self._denoise(x, sigma_vec, x_cond)
            # ODE direction: d = (x - D(x;σ)) / σ  (VE SDE)
            dx = (x - d_cur) / t_cur

            # Euler step
            x_next = x + dt * dx

            # Heun correction (원본과 동일: t_next > 0일 때만)
            if t_next > 0:
                sigma_next_vec = t_next.expand(B)
                d_next = self._denoise(x_next, sigma_next_vec, x_cond)
                dx_next = (x_next - d_next) / t_next
                x = x + dt * (dx + dx_next) * 0.5
            else:
                x = d_cur  # 마지막 step에서는 denoised output 직접 사용

        return x.clamp(-self.clip_x0, self.clip_x0)

    def forward(self, x_cond: torch.Tensor) -> torch.Tensor:
        """Inference: sampling."""
        return self.sample(x_cond, steps=self.sample_steps)


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T = 2, 1000

    model = BeatDiff1to12(
        cond_ch=1, out_ch=12, base_ch=64,
        channel_mult=(1, 2, 3),
        num_blocks=2,
        sample_steps=5, val_sample_steps=5,
    )

    x_cond = torch.randn(B, 1, T)
    y_true = torch.randn(B, 12, T)

    # Training loss
    loss, pred, terms = model.loss_and_pred(x_cond, y_true)
    print(f"Loss: {loss.item():.4f}")
    print(f"Pred shape: {pred.shape}")
    print(f"Terms: {terms}")

    # Sampling
    sampled = model.sample(x_cond, steps=5)
    print(f"Sample shape: {sampled.shape}")

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    print("\n[OK] BeatDiff1to12 test passed!")
