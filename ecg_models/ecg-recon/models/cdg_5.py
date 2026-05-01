# -*- coding: utf-8 -*-
"""
CDGS v5 (`cdg_5.py`): **`CDGS5`** — `cdg_4`의 포크.

동일 아이디어:
  - unit ball 안의 64개 셀(4×4×4 codebook) ↔ 64개 가우시안을 **1:1로 할당**
    (Sinkhorn 연속 → ST-hard 유니크 퍼뮤테이션).
  - 셀 중심 + 셀 내부 연속 delta(t) (shared MLP)로 **상자 안에서만** 3D 이동.
  - 옵션 A 기본: 각도 감쇠(hem, exp(-γθ²))는 끄고 SH에 각도 의존을 맡김.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_CHEST_VECTORS_RAW = [
    [-0.707, 0.000, 0.707],  # V1
    [0.000, 0.000, 1.000],  # V2
    [0.438, 0.000, 0.899],  # V3
    [0.707, 0.000, 0.707],  # V4
    [0.899, 0.000, 0.438],  # V5
    [0.966, 0.000, 0.259],  # V6
]
CHEST_VECTORS_INIT = F.normalize(torch.tensor(_CHEST_VECTORS_RAW, dtype=torch.float32), dim=-1)


def build_codebook_64(device: torch.device, dtype: torch.dtype, *, r_max: float = 0.85) -> torch.Tensor:
    xs = torch.linspace(-1.0, 1.0, 4, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, 4, device=device, dtype=dtype)
    zs = torch.linspace(-1.0, 1.0, 4, device=device, dtype=dtype)
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(-1, 3)  # (64,3)
    nrm = grid.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = torch.minimum(torch.ones_like(nrm), (r_max / nrm))
    return grid * scale


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 30) -> torch.Tensor:
    x = log_alpha - log_alpha.amax(dim=(-2, -1), keepdim=True)
    x = torch.exp(x)
    for _ in range(int(n_iters)):
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)
        x = x / (x.sum(dim=-2, keepdim=True) + 1e-8)
    return x


def greedy_unique_argmax(scores: torch.Tensor) -> torch.Tensor:
    B, N, K = scores.shape
    if N != K:
        raise ValueError(f"unique assignment requires N==K (got {N} vs {K})")
    out = torch.zeros_like(scores)
    for b in range(B):
        used_rows = torch.zeros(N, device=scores.device, dtype=torch.bool)
        used_cols = torch.zeros(K, device=scores.device, dtype=torch.bool)
        flat = scores[b].reshape(-1)
        order = torch.argsort(flat, descending=True)
        for idx in order:
            r = int(idx // K)
            c = int(idx % K)
            if used_rows[r] or used_cols[c]:
                continue
            out[b, r, c] = 1.0
            used_rows[r] = True
            used_cols[c] = True
            if used_rows.all():
                break
    return out


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


class DeltaMLP(nn.Module):
    def __init__(self, code_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(code_dim + 1, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, D = x.shape
        y = self.net(x.reshape(B * N * T, D))
        return y.reshape(B, N, T, 3)


class GaussianPredictor(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_gaussians=64,
        n_heads=8,
        *,
        codebook_size: int = 64,
        hard_positions: bool = True,
        pos_temp_init: float = 1.1,
        delta_code_dim: int = 8,
    ):
        super().__init__()
        self.n_gaussians = int(n_gaussians)
        self.sh_dim = 9
        self.codebook_size = int(codebook_size)
        self.hard_positions = bool(hard_positions)
        self.delta_code_dim = int(delta_code_dim)

        out_dim = 4 + self.sh_dim * 2 + self.codebook_size + self.delta_code_dim
        _gq = torch.randn(1, self.n_gaussians, d_model)
        self.gaussian_queries = nn.Parameter(F.normalize(_gq, dim=-1, eps=1e-6))
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.param_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, out_dim))
        nn.init.zeros_(self.param_head[-1].bias)
        nn.init.normal_(self.param_head[-1].weight, std=0.02)

        self.pos_temp_raw = nn.Parameter(torch.tensor(math.log(math.expm1(float(pos_temp_init)))))

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
        sh_base = params[..., 4 : 4 + self.sh_dim]
        sh_vel = params[..., 4 + self.sh_dim : 4 + self.sh_dim * 2]
        off = 4 + self.sh_dim * 2
        cell_logits = params[..., off : off + self.codebook_size]
        delta_code = params[..., off + self.codebook_size : off + self.codebook_size + self.delta_code_dim]
        temp = (F.softplus(self.pos_temp_raw) + 1e-6).to(dtype=cell_logits.dtype)
        return mu, sigma0, sigma_vel, amplitude, sh_base, sh_vel, cell_logits, delta_code, temp


def camera_plane_basis(L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


class SharedSplatReadout(nn.Module):
    def __init__(self, in_ch: int = 2, hidden: int = 32, grid: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)


class GaussianRenderer(nn.Module):
    def __init__(
        self,
        splat_grid: int = 8,
        cnn_hidden: int = 32,
        lead_halfspace_gate: bool = False,
        direct_lead_sum: bool = False,
        direct_lead_patch: int = 5,
        direct_lead_span: float = 0.55,
        *,
        use_angle_weight: bool = False,
    ):
        super().__init__()
        self.splat_grid = splat_grid
        self.lead_halfspace_gate = bool(lead_halfspace_gate)
        self.direct_lead_sum = bool(direct_lead_sum)
        self.direct_lead_patch = int(direct_lead_patch)
        self.direct_lead_span = float(direct_lead_span)
        self.use_angle_weight = bool(use_angle_weight)

        H = W = splat_grid
        self.register_buffer("grid_u", torch.linspace(-1.0, 1.0, W).view(1, 1, 1, 1, W))
        self.register_buffer("grid_v", torch.linspace(-1.0, 1.0, H).view(1, 1, 1, H, 1))
        self.splat_tau_raw = nn.Parameter(torch.tensor(0.0))
        self.dist_atten_raw = nn.Parameter(torch.tensor(-0.5))

        self.limb_ra = nn.Parameter(torch.zeros(3))
        self.limb_la = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
        self.limb_ll = nn.Parameter(torch.tensor([0.5, math.sqrt(3) / 2, 0.0]))
        self.chest_vectors = nn.Parameter(CHEST_VECTORS_INIT.clone())
        self.shared_splat_readout = SharedSplatReadout(in_ch=2, hidden=cnn_hidden, grid=splat_grid)

    def limb_vectors(self) -> torch.Tensor:
        ra, la, ll = self.limb_ra, self.limb_la, self.limb_ll
        L_I = F.normalize(la - ra, dim=-1, eps=1e-6)
        L_II = F.normalize(ll - ra, dim=-1, eps=1e-6)
        L_III = F.normalize(ll - la, dim=-1, eps=1e-6)
        L_aVR = F.normalize(ra - (la + ll) * 0.5, dim=-1, eps=1e-6)
        L_aVL = F.normalize(la - (ra + ll) * 0.5, dim=-1, eps=1e-6)
        L_aVF = F.normalize(ll - (ra + la) * 0.5, dim=-1, eps=1e-6)
        return torch.stack([L_I, L_II, L_III, L_aVR, L_aVL, L_aVF], dim=0)

    def forward(
        self,
        mu,
        sigma0,
        sigma_vel,
        amplitude,
        sh_base,
        sh_vel,
        cell_logits,
        delta_code,
        delta_mlp: nn.Module,
        codebook,
        pos_temp,
        hard_positions: bool,
        T: int,
        chest_offset=None,
        resp_freq=None,
        resp_vec=None,
    ):
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
        gauss = amplitude.unsqueeze(-1) * torch.exp(-0.5 * (dt / sigma_t) ** 2)
        sh_dynamic = sh_base.unsqueeze(-1) + sh_vel.unsqueeze(-1) * dt.unsqueeze(2)

        logits = cell_logits / (pos_temp + 1e-6)
        w_soft = sinkhorn(logits, n_iters=30)
        if hard_positions:
            w_hard = greedy_unique_argmax(w_soft)
            w = w_hard + (w_soft - w_soft.detach())
        else:
            w = w_soft
        cell_center = torch.einsum("bnk,kc->bnc", w, codebook)

        cell_half = mu.new_tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 7.0]).view(1, 1, 3, 1)
        dt_feat = dt.unsqueeze(2).permute(0, 1, 3, 2)  # (B,N,T,1)
        code_feat = delta_code.unsqueeze(2).expand(-1, -1, T, -1)  # (B,N,T,C)
        mlp_in = torch.cat([code_feat, dt_feat], dim=-1)
        d_raw = delta_mlp(mlp_in)  # (B,N,T,3)
        delta_t = torch.tanh(d_raw).permute(0, 1, 3, 2) * cell_half
        p_pos = cell_center.unsqueeze(-1) + delta_t

        limb = self.limb_vectors()
        chest = F.normalize(self.chest_vectors, dim=-1)
        if chest_offset is not None:
            chest_personalized = F.normalize(chest.unsqueeze(0) + chest_offset, dim=-1)
            lead_vecs = torch.cat([limb.unsqueeze(0).expand(B, -1, -1), chest_personalized], dim=1)
        else:
            lead_vecs = torch.cat([limb, chest], dim=0).unsqueeze(0).expand(B, -1, -1)

        lead_vecs_t = lead_vecs.unsqueeze(-1).expand(-1, -1, -1, T)
        if resp_freq is not None and resp_vec is not None:
            oscillation = resp_vec.view(B, 1, 3, 1) * torch.sin(
                2 * math.pi * resp_freq.view(B, 1, 1, 1) * t.view(1, 1, 1, T)
            )
            lead_vecs_t = F.normalize(lead_vecs_t + oscillation, dim=2)

        e1, e2 = camera_plane_basis(lead_vecs_t)
        u = torch.einsum("bnct,bkct->bnkt", p_pos, e1)
        v = torch.einsum("bnct,bkct->bnkt", p_pos, e2)
        u = torch.tanh(u)
        v = torch.tanh(v)

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
        )
        sh_proj = torch.einsum("bnmt,bkmt->bnkt", sh_dynamic, Y)

        tau = F.softplus(self.splat_tau_raw) + 0.06
        splat_pos = torch.zeros(B, 12, T, H, W, device=mu.device, dtype=mu.dtype)
        splat_sh = torch.zeros(B, 12, T, H, W, device=mu.device, dtype=mu.dtype)

        if self.use_angle_weight:
            gamma = F.softplus(self.dist_atten_raw) + 1e-6
            p_nrm = p_pos.norm(dim=2, keepdim=True).clamp_min(1e-8)
            cos_p_L = torch.einsum("bnct,bkct->bnkt", p_pos, lead_vecs_t) / p_nrm

        for n in range(N):
            u_n = u[:, n]
            v_n = v[:, n]
            g_n = gauss[:, n]
            sh_n = sh_proj[:, n]
            du = u_n.unsqueeze(-1).unsqueeze(-1) - gx
            dv = v_n.unsqueeze(-1).unsqueeze(-1) - gy
            kernel = torch.exp(-0.5 * (du * du + dv * dv) / (tau * tau))
            g_expand = g_n.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

            if self.use_angle_weight:
                cos_n = cos_p_L[:, n]
                if self.lead_halfspace_gate:
                    hem = torch.clamp_min(cos_n, 0.0)
                else:
                    hem = torch.ones_like(cos_n)
                cos_clamped = cos_n.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                theta = torch.acos(cos_clamped)
                dist_atten = torch.exp(-gamma * theta * theta)
                weight = (hem * dist_atten).unsqueeze(-1).unsqueeze(-1)
                splat_pos = splat_pos + g_expand * kernel * weight
                w_sh = (g_n.unsqueeze(1) * sh_n).unsqueeze(-1).unsqueeze(-1)
                splat_sh = splat_sh + w_sh * kernel * weight
            else:
                splat_pos = splat_pos + g_expand * kernel
                w_sh = (g_n.unsqueeze(1) * sh_n).unsqueeze(-1).unsqueeze(-1)
                splat_sh = splat_sh + w_sh * kernel

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


class CDGS5(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_gaussians=64,
        n_encoder_layers=4,
        use_metadata=True,
        lead_halfspace_gate: bool = False,
        direct_lead_sum: bool = False,
        direct_lead_patch: int = 5,
        direct_lead_span: float = 0.55,
        *,
        hard_positions: bool = True,
        use_angle_weight: bool = False,
        delta_code_dim: int = 8,
        delta_mlp_hidden: int = 32,
    ):
        super().__init__()
        self.use_metadata = use_metadata
        self.encoder = MultiScaleEncoder(d_model, n_encoder_layers)
        cb = build_codebook_64(device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("pos_codebook", cb)
        if int(n_gaussians) != int(cb.shape[0]):
            raise ValueError(
                f"cdg_5 requires n_gaussians==codebook_size==64 for 1:1 assignment (got {n_gaussians} vs {int(cb.shape[0])})"
            )
        self.gaussian_predictor = GaussianPredictor(
            d_model,
            n_gaussians,
            hard_positions=hard_positions,
            codebook_size=cb.shape[0],
            delta_code_dim=delta_code_dim,
        )
        self.hard_positions = bool(hard_positions)
        self.delta_mlp = DeltaMLP(code_dim=int(delta_code_dim), hidden=int(delta_mlp_hidden))
        self.meta_conditioner = None  # keep signature compatible; v5 uses same dummy meta if needed upstream
        self.renderer = GaussianRenderer(
            lead_halfspace_gate=lead_halfspace_gate,
            direct_lead_sum=direct_lead_sum,
            direct_lead_patch=direct_lead_patch,
            direct_lead_span=direct_lead_span,
            use_angle_weight=use_angle_weight,
        )
        self.refiner = ResidualRefiner()

    def forward(self, x, age=None, sex=None):
        _, global_feat = self.encoder(x)
        codebook = self.pos_codebook.to(device=x.device, dtype=global_feat.dtype)
        mu, sigma0, sigma_vel, amplitude, sh_base, sh_vel, cell_logits, delta_code, pos_temp = self.gaussian_predictor(
            global_feat
        )
        pred_coarse = self.renderer(
            mu,
            sigma0,
            sigma_vel,
            amplitude,
            sh_base,
            sh_vel,
            cell_logits,
            delta_code,
            self.delta_mlp,
            codebook,
            pos_temp,
            self.hard_positions,
            T=x.size(2),
            chest_offset=None,
            resp_freq=None,
            resp_vec=None,
        )
        pred_fine = self.refiner(x, pred_coarse)
        return pred_fine, pred_coarse, {
            "mu": mu,
            "sigma0": sigma0,
            "sigma_vel": sigma_vel,
            "amplitude": amplitude,
            "cell_logits": cell_logits,
            "delta_code": delta_code,
            "pos_temp": pos_temp,
        }


class CDGS5Loss(nn.Module):
    def __init__(self, w_recon: float = 1.0, w_coarse: float = 0.3, w_morph: float = 0.25, w_freq: float = 0.1):
        super().__init__()
        self.w_recon = w_recon
        self.w_coarse = w_coarse
        self.w_morph = w_morph
        self.w_freq = w_freq

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)

    def frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.float()
        t = target.float()
        return F.l1_loss(torch.fft.rfft(p, dim=-1).abs(), torch.fft.rfft(t, dim=-1).abs())

    def morphology_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + target.abs() / (target.abs().mean(dim=-1, keepdim=True) + 1e-6)
        return (weights * (pred - target).abs()).mean()

    def forward(self, pred_fine: torch.Tensor, pred_coarse: torch.Tensor, target: torch.Tensor, amplitude: torch.Tensor):
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
        losses["total"] = total
        return total, losses


if __name__ == "__main__":
    torch.manual_seed(0)
    m = CDGS5()
    x = torch.randn(1, 1, 128)
    y_f, _, ex = m(x, torch.tensor([0.5]), torch.tensor([1]))
    assert y_f.shape == (1, 12, 128)
    # sanity: unique assignment (rows/cols sum 1) for hard mode
    w_soft = sinkhorn(ex["cell_logits"] / ex["pos_temp"], n_iters=20)
    w_hard = greedy_unique_argmax(w_soft)
    assert float(w_hard.sum(dim=-1).min()) == 1.0 and float(w_hard.sum(dim=-2).min()) == 1.0
    print("ok")

