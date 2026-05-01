# -*- coding: utf-8 -*-
"""
CDGS v8.1 (`cdg_8.py`): 분석 기반 전면 수정 버전
=======================================================
수정 목록:
  [Fix-1] Envelope bias 제거: sigmoid(x - 1.0) → sigmoid(x) + Learnable bias
  [Fix-2] PhysicsGate 추가: Bypass가 Physics를 압도하지 못하도록 균형 학습
  [Fix-3] Missingness Indicator 지원: meta 6차원 (age, sex, h, w, h_missing, w_missing)
  [Fix-4] electrode_offset 범위 확대: 0.1 → 0.25
  [Fix-5] MetadataElectrodeConditioner: missingness flag 반영
  [Fix-6] Bypass 용량 축소 + Physics 강화를 위한 구조 변경
  [Fix-7] weight_decay 제외 파라미터 그룹 분리 헬퍼 함수 추가
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────────────────
def smooth_clamp_min(x, min_val, beta=10.0):
    return min_val + F.softplus(x - min_val, beta=beta)

_ELECTRODE_INIT = [
    [-1.0,  1.0, -0.5],   # RA
    [ 1.0,  1.0, -0.5],   # LA
    [ 0.5, -1.5,  0.5],   # LL
    [-0.2,  0.0,  1.0],   # V1
    [ 0.0,  0.0,  1.2],   # V2
    [ 0.3, -0.1,  1.1],   # V3
    [ 0.6, -0.2,  1.0],   # V4
    [ 0.9, -0.2,  0.8],   # V5
    [ 1.1, -0.2,  0.4],   # V6
]

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


# ──────────────────────────────────────────────────────────────────────────────
# [Fix-7] weight_decay 제외 파라미터 그룹 분리 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def get_param_groups(model: nn.Module, weight_decay: float):
    """
    bias, LayerNorm, BatchNorm 파라미터에는 weight_decay를 적용하지 않음.
    AdamW 사용 시 반드시 이 함수로 파라미터 그룹을 분리할 것.
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (
            param.ndim <= 1 or
            name.endswith(".bias") or
            "norm" in name.lower() or
            "bn" in name.lower()
        )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# [Fix-3, Fix-5] MetadataElectrodeConditioner — Missingness Indicator 지원
# meta 입력 차원: 4 (기존) → 6 (개선: age, sex, h, w, h_missing, w_missing)
# ──────────────────────────────────────────────────────────────────────────────
class MetadataElectrodeConditioner(nn.Module):
    def __init__(self, n_electrodes: int = 9, hidden_dim: int = 64, meta_dim: int = 6):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.meta_dim = meta_dim

        self.age_proj      = nn.Linear(1, hidden_dim // 4)
        self.sex_emb       = nn.Embedding(3, hidden_dim // 4)
        # h, w 각각 값 + missing flag = 2 + 2 = 4차원
        self.hw_proj       = nn.Linear(4, hidden_dim // 2)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_electrodes * 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        """
        meta: [B, 6] → (age_norm, sex, h_norm, w_norm, h_missing_flag, w_missing_flag)
        반환: [B, n_electrodes, 3] 전극 위치 오프셋
        """
        age           = meta[:, 0:1]
        sex           = meta[:, 1].long()
        h_val         = meta[:, 2:3]
        w_val         = meta[:, 3:4]
        h_miss        = meta[:, 4:5]   # [Fix-3] 결측 여부 플래그
        w_miss        = meta[:, 5:6]

        a    = F.gelu(self.age_proj(age))
        s    = self.sex_emb(sex.clamp(0, 2))
        hw_combined = torch.cat([h_val, w_val, h_miss, w_miss], dim=-1)
        h    = F.gelu(self.hw_proj(hw_combined))

        feat    = torch.cat([a, s, h], dim=-1)
        offsets = torch.tanh(self.net(feat)) * 0.25   # [Fix-4] 0.20 → 0.25
        return offsets.view(-1, self.n_electrodes, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder (변경 없음)
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
        feats = [conv(x) for conv in self.convs]
        return F.gelu(self.merge(self.bn(torch.cat(feats, dim=1))))


class MultiScaleEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.stem       = nn.Conv1d(1, d_model, kernel_size=7, padding=3)
        self.bn_stem    = nn.BatchNorm1d(d_model)
        self.cnn_blocks = nn.ModuleList([DilatedConvBlock(d_model, d_model) for _ in range(3)])
        self.cnn_norms  = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(3)])
        self.downsample = nn.Conv1d(d_model, d_model, kernel_size=4, stride=4, padding=0)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8,
                dim_feedforward=d_model * 4, batch_first=True,
                dropout=0.1,
            )
            for _ in range(n_layers)
        ])

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
# [Fix-1] DipolePredictor — Envelope bias 수정 + meta 6차원 지원
# ──────────────────────────────────────────────────────────────────────────────
class DipolePredictor(nn.Module):
    def __init__(self, d_model=256, n_gaussians=64, n_heads=8, use_metadata=True, meta_dim=6):
        super().__init__()
        self.n_gaussians  = int(n_gaussians)
        self.use_metadata = use_metadata
        out_dim = 10

        _gq = torch.randn(1, self.n_gaussians, d_model)
        self.queries   = nn.Parameter(F.normalize(_gq, dim=-1, eps=1e-6))
        _p0 = fibonacci_unit_sphere(self.n_gaussians, device=_gq.device, dtype=_gq.dtype).unsqueeze(0)
        self.p0_anchor = nn.Parameter((_p0 * 0.40).clone())

        self.cross_attn        = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.05)
        self.norm              = nn.LayerNorm(d_model)
        self.temporal_proj     = nn.Linear(d_model, d_model)
        self.global_shift_proj = nn.Linear(d_model, 3)

        if self.use_metadata:
            self.meta_proj = nn.Linear(meta_dim, d_model)

        self.wave_delay_scale = nn.Parameter(torch.tensor(0.1))

        # [Fix-1] envelope에 learnable bias 추가 (초기값 0.0 → 적당한 활성화 보장)
        self.envelope_bias = nn.Parameter(torch.tensor(0.5))

        self.param_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_dim),
        )
        nn.init.normal_(self.param_head[-1].weight, std=0.02)
        bias = torch.zeros(out_dim)
        bias[6:9] = -2.0
        bias[9]   = -2.0
        self.param_head[-1].bias = nn.Parameter(bias)

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor, meta: torch.Tensor = None):
        B = global_feat.size(0)
        kv = global_feat.permute(0, 2, 1)
        q  = self.queries.expand(B, -1, -1)
        out, _ = self.cross_attn(q, kv, kv)
        q_static = self.norm(q + out)

        if self.use_metadata and meta is not None:
            meta_emb = F.gelu(self.meta_proj(meta)).unsqueeze(1)
            q_static = q_static + meta_emb

        params = self.param_head(q_static)

        p0_delta     = torch.tanh(params[..., 0:3]) * 0.2
        global_pooled = global_feat.mean(dim=-1)
        heart_shift  = torch.tanh(self.global_shift_proj(global_pooled)) * 0.3

        p_pos       = self.p0_anchor.expand(B, -1, -1) + p0_delta + heart_shift.unsqueeze(1)
        dipole_dir  = F.normalize(params[..., 3:6], dim=-1, eps=1e-6)
        volume_xyz  = torch.sigmoid(params[..., 6:9]) * 0.5 + 1e-3
        amplitude   = torch.sigmoid(params[..., 9]) + 1e-4

        temporal_w      = self.temporal_proj(q_static)
        temporal_logits = torch.einsum('bnc,bct->bnt', temporal_w, local_feat) / math.sqrt(temporal_w.size(-1))
        B_size, N_size, T_size = temporal_logits.shape

        idx_sa = amplitude.argmax(dim=1, keepdim=True)
        p_sa   = torch.gather(p_pos, 1, idx_sa.unsqueeze(-1).expand(-1, -1, 3))
        dist   = torch.norm(p_pos - p_sa, dim=-1)
        raw_delay = F.softplus(self.wave_delay_scale) * dist
        delay  = torch.clamp(raw_delay, min=0.0, max=0.1)

        base_x  = torch.linspace(-1, 1, T_size, device=p_pos.device).view(1, 1, T_size)
        grid_x  = base_x.expand(B_size, N_size, T_size) - delay.unsqueeze(-1)
        base_y  = torch.linspace(-1, 1, N_size, device=p_pos.device).view(1, N_size, 1)
        grid_y  = base_y.expand(B_size, N_size, T_size)
        grid_2d = torch.stack([grid_x, grid_y], dim=-1)
        img_logits     = temporal_logits.unsqueeze(1)
        shifted_logits = F.grid_sample(img_logits, grid_2d, padding_mode='zeros', align_corners=True).squeeze(1)

        # [Fix-1] bias=-1.0 제거 → learnable bias 사용으로 Envelope 죽음 방지
        envelope_t = torch.sigmoid(shifted_logits + self.envelope_bias)

        return amplitude, p_pos, dipole_dir, volume_xyz, envelope_t


# ──────────────────────────────────────────────────────────────────────────────
# 물리 기반 쌍극자 렌더러 (Fix-4: electrode_offset 범위 확대)
# ──────────────────────────────────────────────────────────────────────────────
class PhysicsRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("electrode_anchor", torch.tensor(_ELECTRODE_INIT, dtype=torch.float32))
        self.electrode_offset = nn.Parameter(torch.zeros(9, 3))

    def forward(self, amplitude, p_pos, dipole_dir, volume_xyz, envelope_t, electrode_offset=None):
        # [Fix-4] 0.1 → 0.25 범위로 확대
        E_base = self.electrode_anchor + torch.tanh(self.electrode_offset) * 0.25

        if electrode_offset is not None:
            E     = E_base.unsqueeze(0) + electrode_offset
            R_vec = E.unsqueeze(1) - p_pos.unsqueeze(2)
        else:
            E     = E_base
            R_vec = E.view(1, 1, 9, 3) - p_pos.unsqueeze(2)

        B, N = amplitude.shape
        max_volume   = volume_xyz[..., 0] * volume_xyz[..., 1] * volume_xyz[..., 2]
        active_vol_t = max_volume.unsqueeze(-1) * envelope_t
        charge_t     = amplitude.unsqueeze(-1) * active_vol_t
        p_t          = dipole_dir.unsqueeze(-1) * charge_t.unsqueeze(2)

        R_dist_sq_raw = torch.sum(R_vec ** 2, dim=-1)
        R_dist_sq     = smooth_clamp_min(R_dist_sq_raw, 0.25)
        R_dist_cube   = R_dist_sq * torch.sqrt(R_dist_sq)

        dot_product = torch.einsum('bndt,bnkd->bnkt', p_t, R_vec)
        phi_n       = dot_product / R_dist_cube.unsqueeze(-1)
        phi         = torch.sum(phi_n, dim=1)

        RA, LA, LL = phi[:, 0], phi[:, 1], phi[:, 2]
        V1_to_V6   = phi[:, 3:9]

        L_I   = LA - RA
        L_II  = LL - RA
        L_III = LL - LA
        aVR   = RA - 0.5 * (LA + LL)
        aVL   = LA - 0.5 * (RA + LL)
        aVF   = LL - 0.5 * (RA + LA)
        WCT   = (RA + LA + LL) / 3.0
        Chest = V1_to_V6 - WCT.unsqueeze(1)

        limb_leads = torch.stack([L_I, L_II, L_III, aVR, aVL, aVF], dim=1)
        v_leads    = torch.cat([limb_leads, Chest], dim=1)
        return v_leads


# ──────────────────────────────────────────────────────────────────────────────
# [Fix-2] PhysicsGate: Physics vs Bypass 균형 학습 게이트
# bypass가 physics를 압도하지 못하도록, 두 스트림을 학습 가능한 게이트로 합산
# ──────────────────────────────────────────────────────────────────────────────
class PhysicsGate(nn.Module):
    """
    Physics 스트림과 Bypass 스트림을 per-lead, per-time으로 믹싱.
    초기화: gate=0.0 → sigmoid(0)=0.5 → 50:50 시작
    """
    def __init__(self, n_leads=12, d_model=256):
        super().__init__()
        # per-lead 게이트 (시간 독립적인 기본 가중치)
        self.gate_bias = nn.Parameter(torch.zeros(1, n_leads, 1))
        # 로컬 피처 기반 동적 게이트
        self.gate_proj = nn.Conv1d(d_model, n_leads, kernel_size=1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, phys_signal, bypass_signal, local_feat):
        dynamic_gate = torch.sigmoid(self.gate_proj(local_feat) + self.gate_bias)
        return dynamic_gate * phys_signal + (1.0 - dynamic_gate) * bypass_signal, dynamic_gate


# ──────────────────────────────────────────────────────────────────────────────
# CDGS8 메인 모델 (모든 수정 통합)
# ──────────────────────────────────────────────────────────────────────────────
class CDGS8(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_gaussians: int = 256,
        n_encoder_layers: int = 4,
        use_metadata: bool = True,
        meta_dim: int = 6,        # [Fix-3] 기본값 6 (missingness indicator 포함)
        **kwargs
    ):
        super().__init__()
        self.use_metadata = use_metadata
        self.meta_dim     = meta_dim

        self.encoder = MultiScaleEncoder(d_model, n_encoder_layers)
        self.gaussian_predictor = DipolePredictor(
            d_model, n_gaussians, use_metadata=use_metadata, meta_dim=meta_dim
        )
        self.renderer = PhysicsRenderer()

        if use_metadata:
            self.meta_electrode_net = MetadataElectrodeConditioner(
                n_electrodes=9, hidden_dim=64, meta_dim=meta_dim  # hidden_dim도 64로 확대
            )

        self.lead_gain = nn.Parameter(torch.ones(1, 12, 1))
        self.lead_bias = nn.Parameter(torch.zeros(1, 12, 1))

        # [Fix-6] Bypass 용량 축소: d_model//4 → d_model//8 (Physics 우위 유도)
        bypass_hidden = max(16, d_model // 8)
        self.artifact_bypass = nn.Sequential(
            nn.Conv1d(d_model, bypass_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(bypass_hidden, 12, kernel_size=1),
        )
        nn.init.zeros_(self.artifact_bypass[-1].weight)
        nn.init.zeros_(self.artifact_bypass[-1].bias)

        # [Fix-2] Physics/Bypass 균형 게이트
        self.physics_gate = PhysicsGate(n_leads=12, d_model=d_model)

    def forward(self, x, meta=None, bypass_alpha: float = 1.0):
        local_feat, global_feat = self.encoder(x)

        amplitude, p_pos, dipole_dir, volume_xyz, envelope_t = \
            self.gaussian_predictor(local_feat, global_feat, meta=meta)

        electrode_offset = None
        if self.use_metadata and meta is not None:
            electrode_offset = self.meta_electrode_net(meta)

        v_leads        = self.renderer(amplitude, p_pos, dipole_dir, volume_xyz, envelope_t, electrode_offset)
        v_leads_scaled = v_leads * self.lead_gain + self.lead_bias

        # [Fix-2] gate로 physics/bypass 혼합 (bypass_alpha로 스케줄링도 유지)
        artifact_noise   = self.artifact_bypass(local_feat) * bypass_alpha
        v_leads_mixed, gate = self.physics_gate(v_leads_scaled, artifact_noise, local_feat)

        return v_leads_mixed, v_leads_mixed, {
            "p_pos":           p_pos,
            "amplitude":       amplitude,
            "envelope_t":      envelope_t,
            "dipole_dir":      dipole_dir,
            "volume_xyz":      volume_xyz,
            "local_feat":      local_feat,
            "v_leads_scaled":  v_leads_scaled,
            "artifact_noise":  artifact_noise,
            "physics_gate":    gate,  # 모니터링용: 1에 가까울수록 Physics 우세
        }


# ──────────────────────────────────────────────────────────────────────────────
# CDGS8Loss (Physics 우세 유도를 위한 gate regularization 추가)
# ──────────────────────────────────────────────────────────────────────────────
class CDGS8Loss(nn.Module):
    def __init__(
        self,
        w_recon=1.0,
        w_freq=0.15,    # 0.1 → 0.15 상향 (파형 형태 복원 강화)
        w_morph=0.15,   # 0.1 → 0.15 상향
        w_sparse=0.00001,
        w_pos=0.1,
        w_gate=0.05,    # [Fix-2] Physics 우세 유도 정규화
    ):
        super().__init__()
        self.w_recon  = w_recon
        self.w_freq   = w_freq
        self.w_morph  = w_morph
        self.w_sparse = w_sparse
        self.w_pos    = w_pos
        self.w_gate   = w_gate

    def forward(self, pred_fine, pred_coarse, target, amplitude=None, envelope_t=None, p_pos=None, gate=None):
        L_recon = F.l1_loss(pred_fine, target)
        L_freq  = F.l1_loss(
            torch.fft.rfft(pred_fine.float(), dim=-1).abs(),
            torch.fft.rfft(target.float(),    dim=-1).abs()
        )

        weights = 1.0 + target.abs() / (target.abs().mean(dim=-1, keepdim=True) + 1e-6)
        L_morph = (weights * (pred_fine - target).abs()).mean()

        L_sparse_amp = amplitude.mean()    if amplitude  is not None else target.new_tensor(0.0)
        L_sparse_env = envelope_t.mean()   if envelope_t is not None else target.new_tensor(0.0)
        L_sparse     = (L_sparse_amp + L_sparse_env) / 2.0

        L_pos = F.relu(p_pos.norm(dim=-1) - 0.9).mean() if p_pos is not None else target.new_tensor(0.0)

        # [Fix-2] gate가 0.5(균형)보다 낮아지면 패널티 → Physics 우세 유도
        if gate is not None:
            L_gate = F.relu(0.5 - gate.mean())
        else:
            L_gate = target.new_tensor(0.0)

        total = (
            self.w_recon  * L_recon  +
            self.w_freq   * L_freq   +
            self.w_morph  * L_morph  +
            self.w_sparse * L_sparse +
            self.w_pos    * L_pos    +
            self.w_gate   * L_gate
        )
        losses = {
            "recon":  L_recon,
            "freq":   L_freq,
            "morph":  L_morph,
            "sparse": L_sparse,
            "pos":    L_pos,
            "gate":   L_gate,
            "total":  total,
        }
        return total, losses


# ──────────────────────────────────────────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 2
    # [Fix-3] meta가 이제 6차원: (age, sex, h, w, h_missing, w_missing)
    m = CDGS8(d_model=128, n_gaussians=64, n_encoder_layers=2, use_metadata=True, meta_dim=6)
    x    = torch.randn(B, 1, 1000)
    meta = torch.tensor([
        [0.45, 0, 0.0,  0.0,  0.0, 0.0],   # h/w 모두 있음
        [0.62, 1, -0.2, 0.5,  1.0, 0.0],   # h 결측, w 있음
    ])

    y_train, _, extras = m(x, meta, bypass_alpha=0.5)
    assert y_train.shape == (B, 12, 1000), f"Shape mismatch: {y_train.shape}"

    loss_fn = CDGS8Loss()
    tgt     = torch.randn(B, 12, 1000)
    loss_val, breakdown = loss_fn(
        y_train, y_train, tgt,
        amplitude=extras["amplitude"],
        envelope_t=extras["envelope_t"],
        p_pos=extras["p_pos"],
        gate=extras["physics_gate"],
    )
    print("Envelope mean:", extras["envelope_t"].mean().item())
    print("Gate mean (1=Physics):", extras["physics_gate"].mean().item())
    print(f"CDGS8 v8.1 Test OK! loss={loss_val.item():.4f}")
    for k, v in breakdown.items():
        print(f"  {k}: {v.item():.4f}")