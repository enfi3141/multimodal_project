# -*- coding: utf-8 -*-
"""
Conditional Diffusion for 1-lead -> 12-lead ECG reconstruction.

Design goals:
- Plug into the existing train_all_models.py pipeline.
- Keep training/inference entrypoints similar to UNet models.
- Use a lightweight 1D U-Net denoiser with timestep conditioning.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather 1D buffer values at timestep indices and reshape for broadcasting."""
    out = a.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        g1 = 8 if in_ch % 8 == 0 else (4 if in_ch % 4 == 0 else (2 if in_ch % 2 == 0 else 1))
        g2 = 8 if out_ch % 8 == 0 else (4 if out_ch % 4 == 0 else (2 if out_ch % 2 == 0 else 1))
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(t_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base_ch: int = 64, t_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.enc1 = ResBlock1D(in_ch, base_ch, t_dim)
        self.down1 = nn.Conv1d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)

        self.enc2 = ResBlock1D(base_ch, base_ch * 2, t_dim)
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)

        self.mid = ResBlock1D(base_ch * 2, base_ch * 4, t_dim)

        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock1D(base_ch * 4, base_ch * 2, t_dim)

        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock1D(base_ch * 2, base_ch, t_dim)

        self.out = nn.Conv1d(base_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        s1 = self.enc1(x, t_emb)
        h = self.down1(s1)

        s2 = self.enc2(h, t_emb)
        h = self.down2(s2)

        h = self.mid(h, t_emb)

        h = self.up2(h)
        h = self.dec2(torch.cat([h, s2], dim=1), t_emb)

        h = self.up1(h)
        h = self.dec1(torch.cat([h, s1], dim=1), t_emb)

        return self.out(h)


class Diffusion1to12(nn.Module):
    """
    Conditional diffusion model for ECG reconstruction.

    Input:  x_cond [B, 1, T]
    Target: y_true [B, 12, T]
    """

    def __init__(
        self,
        cond_ch: int = 1,
        out_ch: int = 12,
        base_ch: int = 64,
        timesteps: int = 200,
        sample_steps: int = 30,
        val_sample_steps: int = 20,
        recon_weight: float = 0.2,
        clip_x0: float = 3.0,
    ):
        super().__init__()
        self.cond_ch = cond_ch
        self.out_ch = out_ch
        self.timesteps = int(timesteps)
        self.sample_steps = int(sample_steps)
        self.val_sample_steps = int(val_sample_steps)
        self.recon_weight = float(recon_weight)
        self.clip_x0 = float(clip_x0)

        self.denoiser = TinyUNet1D(in_ch=out_ch + cond_ch, out_ch=out_ch, base_ch=base_ch, t_dim=128)

        betas = torch.linspace(1e-4, 2e-2, self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _predict_noise(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        denoise_in = torch.cat([x_t, cond], dim=1)
        return self.denoiser(denoise_in, t)

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def _predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        return (
            x_t - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * pred_noise
        ) / _extract(self.sqrt_alphas_cumprod, t, x_t.shape)

    def loss_and_pred(self, x_cond: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b = y_true.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=y_true.device, dtype=torch.long)

        noise = torch.randn_like(y_true)
        x_t = self._q_sample(y_true, t, noise)
        pred_noise = self._predict_noise(x_t, x_cond, t)

        loss_noise = F.mse_loss(pred_noise, noise)
        pred_x0 = self._predict_x0(x_t, t, pred_noise).clamp(-self.clip_x0, self.clip_x0)
        loss_recon = F.l1_loss(pred_x0, y_true)

        total = loss_noise + self.recon_weight * loss_recon
        terms = {
            "noise": loss_noise.detach(),
            "recon": loss_recon.detach(),
            "total": total.detach(),
        }
        return total, pred_x0.detach(), terms

    @torch.no_grad()
    def sample(self, x_cond: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        b, _, sig_len = x_cond.shape
        steps = int(steps or self.sample_steps)
        steps = max(2, min(steps, self.timesteps))

        x_t = torch.randn((b, self.out_ch, sig_len), device=x_cond.device, dtype=x_cond.dtype)
        schedule = torch.linspace(self.timesteps - 1, 0, steps, device=x_cond.device).long()

        for i in range(schedule.numel()):
            t_cur = int(schedule[i].item())
            t_vec = torch.full((b,), t_cur, device=x_cond.device, dtype=torch.long)

            pred_noise = self._predict_noise(x_t, x_cond, t_vec)
            alpha_bar_t = _extract(self.alphas_cumprod, t_vec, x_t.shape)

            x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            x0 = x0.clamp(-self.clip_x0, self.clip_x0)

            if i == schedule.numel() - 1:
                x_t = x0
                break

            t_next = int(schedule[i + 1].item())
            t_next_vec = torch.full((b,), t_next, device=x_cond.device, dtype=torch.long)
            alpha_bar_next = _extract(self.alphas_cumprod, t_next_vec, x_t.shape)

            # Deterministic DDIM-style update (eta=0)
            x_t = torch.sqrt(alpha_bar_next) * x0 + torch.sqrt(1.0 - alpha_bar_next) * pred_noise

        return x_t

    def forward(self, x_cond: torch.Tensor) -> torch.Tensor:
        return self.sample(x_cond, steps=self.sample_steps)
