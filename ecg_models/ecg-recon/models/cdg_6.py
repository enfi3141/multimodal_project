# -*- coding: utf-8 -*-
"""
CDGS v6 (`cdg_6.py`): **`CDGS6`**

요구사항 반영:
- (이전 cdg_4/5의) 64개 칸(codebook/1:1 할당) 제거 → dipole 위치는 연속 3D.
- 리드(카메라) 초기 방향은 세팅하되, 시간에 따라 작은 "학습 가능한" 모션을 유지.
- SH로 만든 방향 패턴 값이 dipole의 거리(||p(t)||)에 따라 감쇠되도록 적용
    (요청: "거리에 따라서 sh에서 나온 값이 줄어들도록").
- 4DGS-style 공간 공분산: 각 가우시안 공분산 Σ = R S S^T R^T (6D rotation + 3D scales)

Docker (레포 루트에서, bash 기준) — **백그라운드 실행**
--------------------------------------------------------
아래 블록을 그대로 복사해서 실행합니다. `-v` **왼쪽** 경로만 본인 서버의 PTB-XL
`ptb-xl/1.0.3` 폴더로 바꾸세요(안에 `records100` 등 있어야 함).
오른쪽 `:/workspace/data/ptb-xl` 은 고정.

① 학습 — 백그라운드 (nohup)
nohup docker run --rm --gpus '"device=6"' \
    -v ~/project_jyu/physionet.org/files/ptb-xl/1.0.3:/workspace/data/ptb-xl \
    -v "$(pwd)":/workspace -w /workspace ecg-1to12 \
    bash -c "python ecg_1to12/train_all_models.py --model cdg_6 --data_dir ./data/ptb-xl --epochs 30 --batch_size 4 --skip_download" \
    > cdg_6_train.log 2>&1 </dev/null &

로그 확인:
    tail -f cdg_6_train.log

② 3D 시각화 (HTML) — 백그라운드 (nohup)
nohup docker run --rm --gpus '"device=0"' \
    -v "$(pwd)":/workspace -w /workspace ecg-1to12 \
    bash -c "python ecg_1to12/visualize_cdgs_gaussians_3d.py --model cdg_6 --checkpoint /workspace/<CHECKPOINT.pt> --html /workspace/viz_cdg6.html" \
    > cdg_6_viz.log 2>&1 </dev/null &

로그 확인:
    tail -f cdg_6_viz.log

③ Windows PowerShell: `$(pwd)` 대신 `-v C:/Users/이름/프로젝트:/workspace`
(nohup 백그라운드는 WSL/Git Bash에서 위 블록 권장)
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# ──────────────────────────────────────────────────────────────────────────────
# Lead vectors init (same as cdg_3)
# ──────────────────────────────────────────────────────────────────────────────
_CHEST_VECTORS_RAW = [
    [-0.707, 0.000, 0.707],  # V1
    [0.000, 0.000, 1.000],  # V2
    [0.438, 0.000, 0.899],  # V3
    [0.707, 0.000, 0.707],  # V4
    [0.899, 0.000, 0.438],  # V5
    [0.966, 0.000, 0.259],  # V6
]
CHEST_VECTORS_INIT = F.normalize(torch.tensor(_CHEST_VECTORS_RAW, dtype=torch.float32), dim=-1)


def fibonacci_unit_sphere(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if n <= 0:
        return torch.zeros(0, 3, device=device, dtype=dtype)
    idx = torch.arange(n, device=device, dtype=dtype)
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    theta = 2.0 * math.pi * idx / golden
    z = 1.0 - (2.0 * idx + 1.0) / float(n)
    z = z.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    pts = torch.stack([x, y, z], dim=-1)
    return F.normalize(pts, dim=-1, eps=1e-6)


def dipole_position_in_unit_ball(p0: torch.Tensor, p_vel: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Legacy linear motion — kept for visualisation backward-compat."""
    # (B,N,3,T)
    p_raw = p0.unsqueeze(-1) + p_vel.unsqueeze(-1) * dt.unsqueeze(2)
    nrm = p_raw.norm(dim=2, keepdim=True).clamp_min(1e-8)
    return (p_raw / nrm) * torch.tanh(nrm)


class DeformationMLP(nn.Module):
    """4DGS-style deformation field: canonical position + time → nonlinear offset.

    Each Gaussian carries a sample-conditioned latent code predicted by the
    encoder.  The MLP maps (canonical_pos, dt, latent) → Δpos, allowing
    arbitrary continuous trajectories instead of the linear p0 + p_vel·dt.
    Output layer is zero-initialised so training starts from the canonical
    position (identity deformation).
    """

    def __init__(self, d_latent: int = 16, d_hidden: int = 64):
        super().__init__()
        # Input: canonical_pos(3) + relative_time(1) + per-gaussian latent(d_latent)
        self.net = nn.Sequential(
            nn.Linear(3 + 1 + d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 3),  # delta_pos
        )
        # Near-zero init: deformation starts close to identity but with
        # tiny perturbations so gradients can flow from the very first step.
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        canonical_pos: torch.Tensor,
        t: torch.Tensor,
        latent: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """
        canonical_pos: (B, N, 3) — anchor positions from GaussianPredictor
        t : (T,) — normalised time grid [0,1]
        latent : (B, N, d_latent) — per-gaussian deformation code
        mu : (B, N) — temporal centres (for relative time dt = t − μ)
        Returns: p_pos (B, N, 3, T) — deformed positions inside unit ball
        """
        B, N, _ = canonical_pos.shape
        T = t.shape[0]
        pos_exp = canonical_pos.unsqueeze(2).expand(-1, -1, T, -1)   # (B,N,T,3)
        dt = (t.view(1, 1, T) - mu.unsqueeze(-1)).unsqueeze(-1)      # (B,N,T,1)
        lat_exp = latent.unsqueeze(2).expand(-1, -1, T, -1)          # (B,N,T,d_lat)
        inp = torch.cat([pos_exp, dt, lat_exp], dim=-1)              # (B,N,T,3+1+d_lat)
        delta = self.net(inp)                                         # (B,N,T,3)
        p_raw = (canonical_pos.unsqueeze(2) + delta).permute(0, 1, 3, 2)  # (B,N,3,T)
        nrm = p_raw.norm(dim=2, keepdim=True).clamp_min(1e-8)
        return (p_raw / nrm) * torch.tanh(nrm)


def camera_plane_basis(L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """L: (B,K,3,T) unit vectors → plane basis e1,e2 (B,K,3,T)."""
    B, K, _, Te = L.shape
    a = L.new_tensor([1.0, 0.0, 0.0]).view(1, 1, 3, 1).expand(B, K, 3, Te)
    e1 = torch.linalg.cross(L, a, dim=2)
    small = e1.norm(dim=2, keepdim=True) < 1e-4
    a2 = L.new_tensor([0.0, 1.0, 0.0]).view(1, 1, 3, 1).expand(B, K, 3, Te)
    e1_alt = torch.linalg.cross(L, a2, dim=2)
    e1 = torch.where(small.expand_as(e1), e1_alt, e1)
    e1 = F.normalize(e1, dim=2, eps=1e-6)
    e2 = F.normalize(torch.linalg.cross(e1, L, dim=2), dim=2, eps=1e-6)
    return e1, e2


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    d6: (..., 6) -> R: (..., 3, 3)
    Zhou et al. 6D rotation representation.
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=1e-6)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3) columns


def _cov2d_mahalanobis_and_norm(
    Sigma2: torch.Tensor,
    du: torch.Tensor,
    dv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stable 2x2 Gaussian density on (u,v)-grid:
      kernel ∝ exp(-0.5 * q) / sqrt(det(Sigma2)),  q = [du,dv]^T Sigma^{-1} [du,dv]
    Sigma2: (B,12,T,2,2) same dtype as du (often fp16 under AMP); work in fp32.
    """
    S = Sigma2.to(dtype=torch.float32)
    S = 0.5 * (S + S.transpose(-1, -2))
    eps_i = 1e-4
    eye_shape = [1] * (S.dim() - 2) + [2, 2]
    S = S + eps_i * torch.eye(2, device=S.device, dtype=torch.float32).view(*eye_shape)
    a = S[..., 0, 0]
    bsym = 0.5 * (S[..., 0, 1] + S[..., 1, 0])
    c = S[..., 1, 1]
    det = (a * c - bsym * bsym).clamp_min(1e-6)
    inv00 = c / det
    inv01 = -bsym / det
    inv11 = a / det
    inv00 = inv00.to(dtype=du.dtype).unsqueeze(-1).unsqueeze(-1)
    inv01 = inv01.to(dtype=du.dtype).unsqueeze(-1).unsqueeze(-1)
    inv11 = inv11.to(dtype=du.dtype).unsqueeze(-1).unsqueeze(-1)
    
    q = inv00 * (du * du) + (2.0 * inv01) * (du * dv) + inv11 * (dv * dv)
    q = q.clamp(max=80.0)
    norm = torch.rsqrt(det).to(dtype=du.dtype).unsqueeze(-1).unsqueeze(-1)
    return q, norm


# ──────────────────────────────────────────────────────────────────────────────
# Encoder (same shape as cdg_3)
# ──────────────────────────────────────────────────────────────────────────────
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        ch_per = out_ch // len(dilations)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_ch, ch_per, kernel_size=5, padding=d * 2, dilation=d) for d in dilations]
        )
        self.bn = nn.BatchNorm1d(ch_per * len(dilations))
        self.merge = nn.Conv1d(ch_per * len(dilations), out_ch, 1)

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        return F.gelu(self.merge(self.bn(torch.cat(feats, dim=1))))


class MultiScaleEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.stem = nn.Conv1d(1, d_model, kernel_size=7, padding=3)
        self.bn_stem = nn.BatchNorm1d(d_model)
        self.cnn_blocks = nn.ModuleList([DilatedConvBlock(d_model, d_model) for _ in range(3)])
        self.cnn_norms = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(3)])
        self.downsample = nn.Conv1d(d_model, d_model, kernel_size=4, stride=4, padding=0)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=8, dim_feedforward=d_model * 4, batch_first=True
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        h = F.gelu(self.bn_stem(self.stem(x)))
        for block, norm in zip(self.cnn_blocks, self.cnn_norms):
            h = h + norm(block(h))
        local_feat = h
        h_seq = self.downsample(h).permute(0, 2, 1)
        for layer in self.transformer_layers:
            h_seq = layer(h_seq)
        global_feat = h_seq.permute(0, 2, 1)
        return local_feat, global_feat


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian predictor: time Gaussian + SH drift + continuous dipole motion
# ──────────────────────────────────────────────────────────────────────────────
class GaussianPredictor(nn.Module):
    def __init__(self, d_model=256, n_gaussians=64, n_heads=8, d_deform_latent=16):
        super().__init__()
        self.n_gaussians = int(n_gaussians)
        self.sh_dim = 9
        self.d_deform_latent = int(d_deform_latent)
        # deform_latent replaces p_vel; scales(3) + rot6d(6) for spatial covariance
        out_dim = 4 + self.sh_dim * 2 + 3 + self.d_deform_latent + 3 + 6  # mu,sig0,sig_vel,amp, sh×2, p0_delta, deform_latent, scales, rot6d

        _gq = torch.randn(1, self.n_gaussians, d_model)
        self.gaussian_queries = nn.Parameter(F.normalize(_gq, dim=-1, eps=1e-6))
        _p0 = fibonacci_unit_sphere(self.n_gaussians, device=_gq.device, dtype=_gq.dtype).unsqueeze(0)
        self.p0_anchor = nn.Parameter((_p0 * 0.60).clone())  # wider initial spread to avoid collapse

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.param_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, out_dim))
        nn.init.normal_(self.param_head[-1].weight, std=0.02)
        # Custom bias init: sigma0 (index 1) starts at 0.5 so that
        # softplus(0.5) ≈ 0.97 — wide temporal Gaussians let gradients flow.
        bias = torch.zeros(out_dim)
        bias[1] = 0.5   # sigma0 → softplus(0.5) ≈ 0.97
        self.param_head[-1].bias = nn.Parameter(bias)

    def forward(self, global_feat: torch.Tensor):
        B = global_feat.size(0)
        kv = global_feat.permute(0, 2, 1)
        q = self.gaussian_queries.expand(B, -1, -1)
        out, _ = self.cross_attn(q, kv, kv)
        q = q + out
        params = self.param_head(q)

        mu = torch.sigmoid(params[..., 0])
        sigma0 = params[..., 1]
        sigma_vel = params[..., 2]
        amplitude = F.softplus(params[..., 3]) + 1e-6
        sh_base = params[..., 4:13]
        sh_vel = params[..., 13:22]
        p0_delta = params[..., 22:25]
        idx = 25
        deform_latent = params[..., idx:idx + self.d_deform_latent]
        idx += self.d_deform_latent
        scales_raw = params[..., idx:idx + 3]
        rot6d = params[..., idx + 3:idx + 9]
        p0 = self.p0_anchor.expand(B, -1, -1) + p0_delta
        # small positive scales; cap to keep projected Σ2 well-conditioned
        scales = (F.softplus(scales_raw) + 1e-3).clamp(max=1.2)
        return mu, sigma0, sigma_vel, amplitude, sh_base, sh_vel, p0, deform_latent, scales, rot6d


class SharedSplatReadout(nn.Module):
    def __init__(self, in_ch: int = 2, hidden: int = 32, grid: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).flatten(1))


# ──────────────────────────────────────────────────────────────────────────────
# Renderer with (a) learnable camera motion, (b) SH distance attenuation
# ──────────────────────────────────────────────────────────────────────────────
class GaussianRenderer(nn.Module):
    def __init__(
        self,
        splat_grid: int = 8,
        cnn_hidden: int = 32,
        direct_lead_sum: bool = False,
        direct_lead_patch: int = 5,
        direct_lead_span: float = 0.55,
        *,
        enable_camera_motion: bool = True,
    ):
        super().__init__()
        self.splat_grid = int(splat_grid)
        self.direct_lead_sum = bool(direct_lead_sum)
        self.direct_lead_patch = int(direct_lead_patch)
        self.direct_lead_span = float(direct_lead_span)
        self.enable_camera_motion = bool(enable_camera_motion)

        H = W = self.splat_grid
        self.register_buffer("grid_u", torch.linspace(-1.0, 1.0, W).view(1, 1, 1, 1, W))
        self.register_buffer("grid_v", torch.linspace(-1.0, 1.0, H).view(1, 1, 1, H, 1))
        self.splat_tau_raw = nn.Parameter(torch.tensor(0.0))

        # Einthoven triangle (learnable RA/LA/LL)
        self.limb_ra = nn.Parameter(torch.zeros(3))
        self.limb_la = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
        self.limb_ll = nn.Parameter(torch.tensor([0.5, math.sqrt(3) / 2, 0.0]))
        self.chest_vectors = nn.Parameter(CHEST_VECTORS_INIT.clone())
        self.shared_splat_readout = SharedSplatReadout(in_ch=2, hidden=cnn_hidden, grid=self.splat_grid)

        # learnable camera motion: L(t) = normalize(L0 + A_sin*sin(2πft) + A_cos*cos(2πft))
        self.cam_freq_raw = nn.Parameter(torch.tensor(-2.0))  # softplus -> small freq
        self.cam_amp_sin = nn.Parameter(torch.zeros(12, 3))
        self.cam_amp_cos = nn.Parameter(torch.zeros(12, 3))

        # distance attenuation for SH: sh_proj *= exp(-beta * ||p||)
        # Start with very weak attenuation so Gaussians can explore freely.
        self.sh_dist_beta_raw = nn.Parameter(torch.tensor(-3.0))

    def limb_vectors(self) -> torch.Tensor:
        ra = self.limb_ra
        la = self.limb_la
        ll = self.limb_ll
        L_I = F.normalize(la - ra, dim=-1, eps=1e-6)
        L_II = F.normalize(ll - ra, dim=-1, eps=1e-6)
        L_III = F.normalize(ll - la, dim=-1, eps=1e-6)
        L_aVR = F.normalize(ra - (la + ll) * 0.5, dim=-1, eps=1e-6)
        L_aVL = F.normalize(la - (ra + ll) * 0.5, dim=-1, eps=1e-6)
        L_aVF = F.normalize(ll - (ra + la) * 0.5, dim=-1, eps=1e-6)
        return torch.stack([L_I, L_II, L_III, L_aVR, L_aVL, L_aVF], dim=0)

    def forward(self, mu, sigma0, sigma_vel, amplitude, sh_base, sh_vel, p_pos, scales, rot6d, T: int):
        B, N = mu.shape
        if self.direct_lead_sum:
            p = max(1, self.direct_lead_patch)
            s = max(self.direct_lead_span, 1e-6)
            H = W = p
            lin = torch.linspace(-s, s, p, device=mu.device, dtype=mu.dtype)
            gx = lin.view(1, 1, 1, 1, p)
            gy = lin.view(1, 1, 1, p, 1)
        else:
            H = W = self.splat_grid
            gx = self.grid_u.to(device=mu.device, dtype=mu.dtype)
            gy = self.grid_v.to(device=mu.device, dtype=mu.dtype)

        t = torch.linspace(0, 1, T, device=mu.device, dtype=mu.dtype).view(1, 1, T)
        dt = t - mu.unsqueeze(-1)
        sigma_t = F.softplus(sigma0.unsqueeze(-1) + sigma_vel.unsqueeze(-1) * dt) + 1e-3
        g_t = amplitude.unsqueeze(-1) * torch.exp(-0.5 * (dt / sigma_t) ** 2)  # (B,N,T)

        sh_dynamic = sh_base.unsqueeze(-1) + sh_vel.unsqueeze(-1) * dt.unsqueeze(2)  # (B,N,9,T)
        # p_pos is already provided: (B,N,3,T) — deformed by DeformationMLP
        R = rotation_6d_to_matrix(rot6d)  # (B,N,3,3)
        limb = self.limb_vectors()
        chest = F.normalize(self.chest_vectors, dim=-1)
        lead_vecs = torch.cat([limb, chest], dim=0).unsqueeze(0).expand(B, -1, -1)  # (B,12,3)
        lead_vecs_t = lead_vecs.unsqueeze(-1).expand(-1, -1, -1, T)  # (B,12,3,T)

        if self.enable_camera_motion:
            freq = F.softplus(self.cam_freq_raw) + 1e-6
            phase = 2 * math.pi * freq * t  # (1,1,T)
            motion = (
                self.cam_amp_sin.view(1, 12, 3, 1) * torch.sin(phase.view(1, 1, 1, T))
                + self.cam_amp_cos.view(1, 12, 3, 1) * torch.cos(phase.view(1, 1, 1, T))
            )
            lead_vecs_t = F.normalize(lead_vecs_t + motion, dim=2, eps=1e-6)

        e1, e2 = camera_plane_basis(lead_vecs_t)
        u_raw = torch.einsum("bnct,bkct->bnkt", p_pos, e1)
        v_raw = torch.einsum("bnct,bkct->bnkt", p_pos, e2)
        u = torch.tanh(u_raw)
        v = torch.tanh(v_raw)

        # SH basis from lead direction
        x_lv, y_lv, z_lv = lead_vecs_t[:, :, 0, :], lead_vecs_t[:, :, 1, :], lead_vecs_t[:, :, 2, :]
        Y = torch.stack(
            [
                0.282095 * torch.ones_like(x_lv),
                -0.488603 * y_lv,
                0.488603 * z_lv,
                -0.488603 * x_lv,
                1.092548 * x_lv * y_lv,
                -1.092548 * y_lv * z_lv,
                0.315392 * (3.0 * z_lv**2 - 1.0),
                -1.092548 * x_lv * z_lv,
                0.546274 * (x_lv**2 - y_lv**2),
            ],
            dim=2,
        )  # (B,12,9,T)
        sh_proj = torch.einsum("bnmt,bkmt->bnkt", sh_dynamic, Y)  # (B,N,12,T)

        # distance attenuation: camera-projected distance sqrt(u²+v²) per lead
        beta = F.softplus(self.sh_dist_beta_raw) + 1e-6
        proj_dist = torch.sqrt(u_raw * u_raw + v_raw * v_raw + 1e-8)  # (B,N,12,T)
        sh_atten = torch.exp(-beta * proj_dist)  # (B,N,12,T) — per lead
        sh_proj = sh_proj * sh_atten

        tau = F.softplus(self.splat_tau_raw) + 0.06
        splat_pos = torch.zeros(B, 12, T, H, W, device=mu.device, dtype=mu.dtype)
        splat_sh = torch.zeros(B, 12, T, H, W, device=mu.device, dtype=mu.dtype)

        E_bt = torch.stack([e1, e2], dim=2).permute(0, 1, 4, 2, 3).unsqueeze(1)

        def _render_chunk(R_ch, scales_ch, u_ch, v_ch, g_t_ch, sh_proj_ch, E_bt_tensor, gx_tensor, gy_tensor, tau_tensor):
            s2_chunk = (scales_ch ** 2).clamp_min(1e-8)
            Sigma3_chunk = R_ch @ torch.diag_embed(s2_chunk) @ R_ch.transpose(-1, -2)
            Sigma3_bt_chunk = Sigma3_chunk.unsqueeze(2).unsqueeze(2)

            Sigma2_chunk = E_bt_tensor @ Sigma3_bt_chunk @ E_bt_tensor.transpose(-1, -2)
            I2_tensor = torch.eye(2, device=Sigma2_chunk.device, dtype=Sigma2_chunk.dtype).view(1, 1, 1, 1, 2, 2)
            Sigma2_chunk = Sigma2_chunk + (tau_tensor * tau_tensor) * I2_tensor

            du_chunk = u_ch.unsqueeze(-1).unsqueeze(-1) - gx_tensor.unsqueeze(1)
            dv_chunk = v_ch.unsqueeze(-1).unsqueeze(-1) - gy_tensor.unsqueeze(1)

            q_chunk, norm_chunk = _cov2d_mahalanobis_and_norm(Sigma2_chunk, du_chunk, dv_chunk)
            kernel_chunk = torch.exp(-0.5 * q_chunk) * norm_chunk

            g_expand_chunk = g_t_ch.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
            w_sh_chunk = (g_t_ch.unsqueeze(2) * sh_proj_ch).unsqueeze(-1).unsqueeze(-1)

            pos_diff = (g_expand_chunk * kernel_chunk).sum(dim=1)
            sh_diff = (w_sh_chunk * kernel_chunk).sum(dim=1)
            return pos_diff, sh_diff

        # Chunk the N dimension to support very large N (e.g. 256 or 2048)
        # Using Gradient Checkpointing prevents OOM by clearing intermediate chunk activations.
        # We set chunk_size very small (8) to ensure the peak memory per chunk during the backward pass is <500MB.
        chunk_size = 8
        for i in range(0, N, chunk_size):
            end = min(N, i + chunk_size)
            R_chunk = R[:, i:end]
            scales_chunk = scales[:, i:end]
            u_chunk = u[:, i:end]
            v_chunk = v[:, i:end]
            g_t_chunk = g_t[:, i:end]
            sh_proj_chunk = sh_proj[:, i:end]

            if self.training:
                pos_diff, sh_diff = torch.utils.checkpoint.checkpoint(
                    _render_chunk,
                    R_chunk, scales_chunk, u_chunk, v_chunk, g_t_chunk, sh_proj_chunk, E_bt, gx, gy, tau,
                    use_reentrant=False
                )
            else:
                pos_diff, sh_diff = _render_chunk(
                    R_chunk, scales_chunk, u_chunk, v_chunk, g_t_chunk, sh_proj_chunk, E_bt, gx, gy, tau
                )

            splat_pos = splat_pos + pos_diff
            splat_sh = splat_sh + sh_diff

        outs = []
        for k in range(12):
            inp = torch.stack([splat_pos[:, k], splat_sh[:, k]], dim=2).reshape(B * T, 2, H, W)
            outs.append(self.shared_splat_readout(inp))
        return torch.cat(outs, dim=1).view(B, 12, T)


class ResidualRefiner(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(13, base_ch, 3, padding=1), nn.GELU(), nn.Conv1d(base_ch, 12, 1))

    def forward(self, lead_I, coarse):
        return coarse + self.net(torch.cat([lead_I, coarse], dim=1))


class CDGS6(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_gaussians=64,
        n_encoder_layers=4,
        use_metadata: bool = False,  # v6: metadata unused
        direct_lead_sum: bool = False,
        direct_lead_patch: int = 5,
        direct_lead_span: float = 0.55,
        *,
        enable_camera_motion: bool = True,
        d_deform_latent: int = 16,
        d_deform_hidden: int = 64,
    ):
        super().__init__()
        self.use_metadata = bool(use_metadata)
        self.encoder = MultiScaleEncoder(d_model, n_encoder_layers)
        self.gaussian_predictor = GaussianPredictor(d_model, n_gaussians, d_deform_latent=d_deform_latent)
        self.deform_mlp = DeformationMLP(d_latent=d_deform_latent, d_hidden=d_deform_hidden)
        self.renderer = GaussianRenderer(
            direct_lead_sum=direct_lead_sum,
            direct_lead_patch=direct_lead_patch,
            direct_lead_span=direct_lead_span,
            enable_camera_motion=enable_camera_motion,
        )
        self.refiner = ResidualRefiner()

    def forward(self, x, age=None, sex=None):
        del age, sex
        _, global_feat = self.encoder(x)
        mu, sigma0, sigma_vel, amplitude, sh_base, sh_vel, p0, deform_latent, scales, rot6d = self.gaussian_predictor(global_feat)
        # 4DGS-style: DeformationMLP produces nonlinear trajectories
        T = x.size(2)
        t_grid = torch.linspace(0, 1, T, device=x.device, dtype=torch.float32)
        p_pos = self.deform_mlp(p0.float(), t_grid, deform_latent.float(), mu.float())
        # 학습 시 autocast(FP16)가 켜져 있어도, 이방 공분산 스플랫은 FP32로만 계산 (NaN/불안정 방지).
        amp_off = torch.cuda.amp.autocast(enabled=False) if x.is_cuda else nullcontext()
        with amp_off:
            pred_coarse = self.renderer(
                mu.float(),
                sigma0.float(),
                sigma_vel.float(),
                amplitude.float(),
                sh_base.float(),
                sh_vel.float(),
                p_pos,
                scales.float(),
                rot6d.float(),
                T=T,
            )
        pred_coarse = torch.nan_to_num(pred_coarse, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=x.dtype)
        pred_fine = self.refiner(x, pred_coarse)
        return pred_fine, pred_coarse, {
            "mu": mu,
            "sigma0": sigma0,
            "sigma_vel": sigma_vel,
            "amplitude": amplitude,
            "p0": p0,
            "p_pos": p_pos,
            "scales": scales,
            "rot6d": rot6d,
        }


class CDGS6Loss(nn.Module):
    """CDGS3Loss + diversity regularisation to prevent Gaussian collapse."""

    def __init__(
        self,
        w_recon: float = 1.0,
        w_coarse: float = 0.3,
        w_morph: float = 0.25,
        w_freq: float = 0.1,
        w_diversity: float = 0.05,
    ):
        super().__init__()
        self.w_recon = w_recon
        self.w_coarse = w_coarse
        self.w_morph = w_morph
        self.w_freq = w_freq
        self.w_diversity = w_diversity

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)

    def frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.float()
        t = target.float()
        return F.l1_loss(torch.fft.rfft(p, dim=-1).abs(), torch.fft.rfft(t, dim=-1).abs())

    def morphology_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + target.abs() / (target.abs().mean(dim=-1, keepdim=True) + 1e-6)
        return (weights * (pred - target).abs()).mean()

    @staticmethod
    def diversity_loss(p_pos: torch.Tensor) -> torch.Tensor:
        """Penalise Gaussian co-location to prevent mode collapse.

        Uses the mean position over time for each Gaussian, then
        encourages all pairwise distances to stay above a minimum
        threshold via a soft hinge.

        p_pos: (B, N, 3, T)
        """
        # mean position over time: (B, N, 3)
        mean_pos = p_pos.mean(dim=-1)
        # pairwise squared distance: (B, N, N)
        diff = mean_pos.unsqueeze(2) - mean_pos.unsqueeze(1)  # (B,N,N,3)
        dist_sq = (diff * diff).sum(dim=-1)  # (B,N,N)
        # soft hinge: penalise pairs closer than margin
        margin = 0.05  # squared-distance margin (~0.22 Euclidean)
        penalty = F.relu(margin - dist_sq)  # (B,N,N)
        # exclude diagonal
        N = mean_pos.shape[1]
        mask = 1.0 - torch.eye(N, device=penalty.device, dtype=penalty.dtype).unsqueeze(0)
        return (penalty * mask).sum() / (mask.sum() * mean_pos.shape[0] + 1e-8)

    def forward(
        self,
        pred_fine: torch.Tensor,
        pred_coarse: torch.Tensor,
        target: torch.Tensor,
        amplitude: torch.Tensor,
        p_pos: torch.Tensor | None = None,
    ):
        del amplitude
        losses = {
            "recon": self.reconstruction_loss(pred_fine, target),
            "coarse": self.reconstruction_loss(pred_coarse, target),
            "morph": self.morphology_loss(pred_fine, target),
            "freq": self.frequency_loss(pred_fine, target),
        }
        total = (
            self.w_recon * losses["recon"]
            + self.w_coarse * losses["coarse"]
            + self.w_morph * losses["morph"]
            + self.w_freq * losses["freq"]
        )
        if p_pos is not None and self.w_diversity > 0:
            div = self.diversity_loss(p_pos)
            losses["diversity"] = div
            total = total + self.w_diversity * div
        losses["total"] = total
        return total, losses


if __name__ == "__main__":
    torch.manual_seed(0)
    m = CDGS6()
    x = torch.randn(1, 1, 128)
    y, _, _ = m(x)
    assert y.shape == (1, 12, 128)
    print("ok")

