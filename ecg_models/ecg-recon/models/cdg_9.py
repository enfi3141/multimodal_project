# -*- coding: utf-8 -*-
"""
CDGS v9.5 (`cdg_9.py`): Stability Fix & Physics Gradient Recovery
=======================================================
[v9.5 수정사항]
  1. L_tconc / L_planarity 벡터화: for 루프 + .item() 제거 → CUDA 동기화 해소
  2. L_amp_sparse → L_amp_alive로 변경: amplitude가 죽지 않도록 "살리는" loss
  3. soft_count 정규화 제거: VCG 신호 소멸 방지
  4. Gate 초기값 0.0 (sigmoid=0.5): physics/bypass 50:50 시작
  5. L_tconc, L_planarity를 순수 텐서 연산으로 변경 (gradient 유지)
  6. bare except 제거 → SVD 실패 시 안전한 fallback
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Dower 역변환 행렬
# ──────────────────────────────────────────────────────────────────────────────
_DOWER_8x3 = torch.tensor([
    [-0.172, -0.074,  0.122,  0.231,  0.239,  0.194,  0.156, -0.010],
    [ 0.057, -0.019, -0.106, -0.022,  0.041,  0.048,  0.062,  0.136],
    [-0.229, -0.310,  0.022, -0.011,  0.174,  0.191,  0.177,  0.032],
]).T

def get_dower_12x3() -> torch.Tensor:
    D = _DOWER_8x3
    row_I   = D[0]
    row_II  = D[1]
    row_III = row_II - row_I
    row_aVR = -0.5 * (row_I + row_II)
    row_aVL = row_I  - 0.5 * row_II
    row_aVF = row_II - 0.5 * row_I

    return torch.stack([
        row_I, row_II, row_III, row_aVR, row_aVL, row_aVF,
        D[2], D[3], D[4], D[5], D[6], D[7],
    ])

def smooth_clamp_min(x, min_val, beta=10.0):
    return min_val + F.softplus(x - min_val, beta=beta)

# ------------------ ▼ 여기부터 복사해서 추가 ▼ ------------------
def get_param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

# ──────────────────────────────────────────────────────────────────────────────
# Metadata Conditioner
# ──────────────────────────────────────────────────────────────────────────────
class MetaConditioner(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.age_proj = nn.Linear(1, 8)
        self.sex_emb  = nn.Embedding(3, 8)
        self.hw_proj  = nn.Linear(2, 16)
        
        self.net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(32, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 12 * 3)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        age = meta[:, 0:1]
        sex = meta[:, 1].long().clamp(0, 2)
        hw  = meta[:, 2:4]

        a = F.gelu(self.age_proj(age))
        s = self.sex_emb(sex)
        h = F.gelu(self.hw_proj(hw))
        
        feat = torch.cat([a, s, h], dim=-1)
        d_adj = self.net(feat).view(-1, 12, 3)
        return torch.tanh(d_adj) * 0.2

# ──────────────────────────────────────────────────────────────────────────────
# Encoder (Lead Identity 적용)
# ──────────────────────────────────────────────────────────────────────────────
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        ch_per = out_ch // len(dilations)
        channels = [ch_per] * len(dilations)
        channels[-1] += out_ch % len(dilations)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, c, kernel_size=5, padding=d * 2, dilation=d)
            for c, d in zip(channels, dilations)
        ])
        self.bn    = nn.BatchNorm1d(out_ch)
        self.merge = nn.Conv1d(out_ch, out_ch, 1)

    def forward(self, x):
        return F.gelu(self.merge(self.bn(torch.cat([c(x) for c in self.convs], dim=1))))

class BeatAlignedStem(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.stem    = nn.Conv1d(2, d_model, kernel_size=7, padding=3)
        self.bn_stem = nn.BatchNorm1d(d_model)

    def _r_energy(self, x: torch.Tensor) -> torch.Tensor:
        diff = torch.diff(x, dim=-1, prepend=x[:, :, :1])
        sq   = diff ** 2
        k    = torch.ones(1, 1, 31, device=x.device, dtype=x.dtype) / 31.0
        eng  = F.conv1d(sq, k, padding=15)[:, :, :x.size(-1)]
        return eng / (eng.amax(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r_ch = self._r_energy(x)
        return F.gelu(self.bn_stem(self.stem(torch.cat([x, r_ch], dim=1))))

class MultiScaleEncoder(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 4):
        super().__init__()
        self.beat_stem  = BeatAlignedStem(d_model)
        self.lead_embed = nn.Embedding(12, d_model)
        self.cnn_blocks = nn.ModuleList([DilatedConvBlock(d_model, d_model) for _ in range(3)])
        self.cnn_norms  = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(3)])
        self.downsample = nn.Conv1d(d_model, d_model, kernel_size=4, stride=4, padding=0)
        self.tf_layers  = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model * 4, batch_first=True, dropout=0.1) 
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, lead_id: torch.Tensor = None):
        h = self.beat_stem(x)
        if lead_id is not None:
            h = h + self.lead_embed(lead_id).unsqueeze(-1)
        for block, norm in zip(self.cnn_blocks, self.cnn_norms):
            h = h + norm(block(h))
        local_feat = h
        h_seq = self.downsample(h).permute(0, 2, 1)
        for layer in self.tf_layers:
            h_seq = layer(h_seq)
        return local_feat, h_seq.permute(0, 2, 1)

# ──────────────────────────────────────────────────────────────────────────────
# DipolePredictor (8D 확장 & 활성화 구역 분리)
# [v9.5] amplitude 초기값 수정: bias=0.0 → sigmoid(0)=0.5 시작 (약한 활성화)
# ──────────────────────────────────────────────────────────────────────────────
class DipolePredictor(nn.Module):
    def __init__(self, d_model: int = 256, n_gaussians: int = 512, n_heads: int = 8):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.queries      = nn.Parameter(F.normalize(torch.randn(1, n_gaussians, d_model), dim=-1))
        self.cross_attn   = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.05)
        self.norm         = nn.LayerNorm(d_model)
        self.temporal_proj = nn.Linear(d_model, d_model)
        self.envelope_bias = nn.Parameter(torch.tensor(0.5))

        # 8D 쌍극자 방향 + 1D 진폭 = 9 차원
        self.param_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 9),
        )
        
        # [v9.5] amplitude bias를 -1.0으로 변경 (sigmoid(-1)≈0.27, 죽지 않는 적당한 초기값)
        nn.init.normal_(self.param_head[-1].weight, std=0.01)
        bias = torch.zeros(9)
        bias[8] = -1.0  # sigmoid(-1.0) ≈ 0.27: 적당히 sparse하면서 gradient 유지
        self.param_head[-1].bias = nn.Parameter(bias)
        
        # 가우시안별 4개 해부학적 구역(Region) 소속 확률
        self.region_logits = nn.Parameter(torch.randn(n_gaussians, 4))

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor):
        B = global_feat.size(0)
        q  = self.queries.expand(B, -1, -1)
        kv = global_feat.permute(0, 2, 1)
        out, _ = self.cross_attn(q, kv, kv)
        q_static = self.norm(q + out)

        params      = self.param_head(q_static)
        dipole_dir8 = F.normalize(params[..., 0:8], dim=-1, eps=1e-6) # 8D 차원
        amplitude   = torch.sigmoid(params[..., 8]) + 1e-4

        temp_w          = self.temporal_proj(q_static)
        temporal_logits = torch.einsum('bnc,bct->bnt', temp_w, local_feat) / math.sqrt(temp_w.size(-1))
        envelope_t      = torch.sigmoid(temporal_logits + self.envelope_bias)

        return amplitude, dipole_dir8, envelope_t, self.region_logits

# ──────────────────────────────────────────────────────────────────────────────
# VCGRenderer8D (8D 공간 확장)
# [v9.5] soft_count 정규화 제거 → 단순 합산 (amplitude가 살아있으므로 자연스럽게 스케일)
# ──────────────────────────────────────────────────────────────────────────────
class VCGRenderer8D(nn.Module):
    def __init__(self, n_gaussians: int = 512):
        super().__init__()
        self.n_gaussians = n_gaussians
        
        # 3D Dower (물리 제약) + 5D Extra (비쌍극자 보정 공간)
        self.register_buffer("D_dower", get_dower_12x3())   # [12, 3]
        self.D_extra = nn.Parameter(torch.randn(12, 5) * 0.01)  # [12, 5]
        
        self.vcg_scale = nn.Parameter(torch.ones(8))

    def forward(self, amplitude, dipole_dir8, envelope_t, D_adj=None):
        charge = amplitude.unsqueeze(-1) * envelope_t
        p_t    = dipole_dir8.unsqueeze(-1) * charge.unsqueeze(2)  # [B, N, 8, T]
        
        # [v9.5] soft_count 정규화 제거 → 단순 합산 후 스케일만 적용
        vcg8 = p_t.sum(dim=1)  # [B, 8, T]
        vcg8 = vcg8 * F.softplus(self.vcg_scale).unsqueeze(0).unsqueeze(-1)
        
        D_dower_adj = self.D_dower.unsqueeze(0) # [1, 12, 3]
        if D_adj is not None:
            D_dower_adj = D_dower_adj + D_adj   # [B, 12, 3]
            
        D_extra_exp = self.D_extra.unsqueeze(0).expand(D_dower_adj.size(0), -1, -1) # [B, 12, 5]
        D_full = torch.cat([D_dower_adj, D_extra_exp], dim=-1)  # [B, 12, 8]
        
        v_leads = torch.einsum('bld,bdt->blt', D_full, vcg8)
        
        # 물리적 로깅 및 평면성 Loss를 위해 순수 3D VCG만 반환
        return v_leads, vcg8[:, :3, :]

# ──────────────────────────────────────────────────────────────────────────────
# Residual Bypass
# ──────────────────────────────────────────────────────────────────────────────
class ResidualBypass(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        hidden = d_model // 4  
        self.net = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, dilation=1), 
            nn.GELU(),
            nn.Conv1d(hidden, 12, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, local_feat: torch.Tensor) -> torch.Tensor:
        return self.net(local_feat)

# ──────────────────────────────────────────────────────────────────────────────
# CDGS9 메인 모델
# [v9.5] Gate 초기값 0.0 (sigmoid=0.5) → physics/bypass 50:50 시작
# ──────────────────────────────────────────────────────────────────────────────
class CDGS9(nn.Module):
    def __init__(self, d_model: int = 256, n_gaussians: int = 512, n_encoder_layers: int = 4, **kwargs):
        super().__init__()
        self.encoder   = MultiScaleEncoder(d_model, n_encoder_layers)
        self.predictor = DipolePredictor(d_model, n_gaussians)
        self.renderer  = VCGRenderer8D(n_gaussians)
        self.bypass    = ResidualBypass(d_model)
        self.meta_net  = MetaConditioner()

        self.lead_gain = nn.Parameter(torch.ones(1, 12, 1))
        self.lead_bias = nn.Parameter(torch.zeros(1, 12, 1))
        
        # [v9.5] Gate 초기값 0.0 → sigmoid(0)=0.5: Physics 50% / Bypass 50% 시작
        self.gate = nn.Parameter(torch.full((1, 12, 1), 0.0))

    def forward(self, x, meta=None, bypass_alpha: float = 1.0, lead_id: torch.Tensor = None):
        local_feat, global_feat = self.encoder(x, lead_id=lead_id)

        D_adjust_meta = None
        if meta is not None:
            D_adjust_meta = self.meta_net(meta)

        amplitude, dipole_dir8, envelope_t, region_logits = self.predictor(local_feat, global_feat)
        v_leads, vcg3d = self.renderer(amplitude, dipole_dir8, envelope_t, D_adjust_meta)
        phys_out  = v_leads * self.lead_gain + self.lead_bias

        bypass_out = self.bypass(local_feat) * bypass_alpha

        gate_w    = torch.sigmoid(self.gate)
        mixed_out = gate_w * phys_out + (1.0 - gate_w) * bypass_out

        return mixed_out, phys_out, {
            "amplitude":     amplitude,
            "envelope_t":    envelope_t,
            "vcg":           vcg3d,
            "gate":          gate_w,
            "region_logits": region_logits,
        }

# ──────────────────────────────────────────────────────────────────────────────
# CDGS9Loss (v9.5: 크래시 유발 for 루프 제거 & amplitude 생존 보장)
# ──────────────────────────────────────────────────────────────────────────────
def pearson_loss(pred, target):
    pred_c   = pred   - pred.mean(dim=-1, keepdim=True)
    target_c = target - target.mean(dim=-1, keepdim=True)
    num  = (pred_c * target_c).sum(dim=-1)
    denom = pred_c.norm(dim=-1) * target_c.norm(dim=-1) + 1e-8
    return 1.0 - (num / denom).mean()

class CDGS9Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_recon     = 1.0
        self.w_pearson   = 0.3
        self.w_freq      = 0.1
        self.w_amp_alive = 0.01

    def forward(self, pred, target, amplitude, envelope_t=None, vcg=None, gate=None, region_logits=None):
        # 1. Base reconstruction (해상도 무관)
        L_recon   = F.l1_loss(pred, target)
        L_pearson = pearson_loss(pred, target)
        L_freq    = F.l1_loss(
            torch.fft.rfft(pred.float(), dim=-1).abs(),
            torch.fft.rfft(target.float(), dim=-1).abs(),
        )

        # 2. Amplitude 생존: amplitude가 죽지 않도록
        amp_mean    = amplitude.mean()
        L_amp_alive = F.relu(0.1 - amp_mean)

        total = (
            self.w_recon     * L_recon     +
            self.w_pearson   * L_pearson   +
            self.w_freq      * L_freq      +
            self.w_amp_alive * L_amp_alive
        )

        return total, {
            "recon":     L_recon,
            "pearson":   L_pearson,
            "freq":      L_freq,
            "amp_alive": L_amp_alive,
            "total":     total,
        }

if __name__ == "__main__":
    torch.manual_seed(0)
    B = 2
    m = CDGS9(d_model=128, n_gaussians=256, n_encoder_layers=2)
    x = torch.randn(B, 1, 1000)
    lead_id = torch.tensor([0, 1])

    y_train, y_phys, extras = m(x, bypass_alpha=0.5, lead_id=lead_id)
    assert y_train.shape == (B, 12, 1000)

    loss_fn = CDGS9Loss()
    tgt     = torch.randn(B, 12, 1000)
    loss_val, bd = loss_fn(
        y_train, tgt,
        extras["amplitude"],
    )
    print("\n[OK] CDGS9 (v10 단순화) Test 통과!")
    print(f"  Total Loss: {loss_val.item():.4f}")
    print(f"  Loss terms: {list(bd.keys())}")
    print(f"  Amplitude mean: {extras['amplitude'].mean().item():.4f}")
    print(f"  Gate mean: {extras['gate'].mean().item():.4f}")