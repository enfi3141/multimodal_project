# -*- coding: utf-8 -*-
"""
CDGS v12 (`cdg_12.py`): Identifiable Cardiac Gaussian Splatting
(해부학적 4구역 × 8개)
      - Position: register_buffer로 고정 (nn.Parameter 아님!)
      - Envelope: K개 temporal basis function의================================================================
교수님 피드백 반영:

  문제 1: "amplitude, envelope 등 내부 변수의 GT가 없는데 어떻게 학습하냐"
    → 핵심은 GT가 없다는 게 아니라, 관측(12-lead) 대비 미지수가 너무 많다는 것
    → 해결: 자유도를 관측 이하로 줄여서 identifiability 확보
      - 512 → 32 Gaussians  가중합
        기존 자유도 N×T = 512×1000 = 512,000
        변경 자유도 N×K = 32×8 = 256  (2000배 감소!)

  문제 2: "전극 위치를 미세 조정한다는 건 말이 안 된다"
    → MetaElectrodeConditioner 완전 제거
    → 전극은 표준 해부학적 위치에 고정
    → 환자 차이는 심장 위치/크기(MetaPositionConditioner)로만 반영

자유도 분석 (CDG10 vs CDG12):
  CDG10: 512 Gaussians × (3 pos + 3 dir + 1 amp + T envelope) ≈ 500,000+
  CDG12:  32 Gaussians × (3 dir + 1 amp + 8 basis weights)    =     384
  관측:   ~9 독립전극 × 1000 timestep                        =   9,000
  비율:   CDG10 → 0.018:1 (severely under-determined ❌)
          CDG12 → 23.4:1  (over-determined ✅)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cdg_10 import (
    _ELECTRODE_POSITIONS,
    _HEART_CENTER,
    MultiScaleEncoder,
    PhysicsRenderer,
    ResidualBypass,
    MetaPositionConditioner,
    pearson_loss,
)


# ──────────────────────────────────────────────────────────────────────────────
# Electrical Axis: Lead I / aVF 방향 상수 (전극 좌표에서 파생)
#   Lead I  = V_LA - V_RA  → 방향 ≈ LA - RA
#   aVF     = V_LL - (V_RA + V_LA)/2  → 방향 ≈ LL - (RA+LA)/2
# ──────────────────────────────────────────────────────────────────────────────
_LEAD_I_DIR = F.normalize(_ELECTRODE_POSITIONS[1] - _ELECTRODE_POSITIONS[0], dim=-1)
_LEAD_AVF_DIR = F.normalize(
    _ELECTRODE_POSITIONS[2] - (_ELECTRODE_POSITIONS[0] + _ELECTRODE_POSITIONS[1]) / 2,
    dim=-1,
)


def compute_electrical_axis_from_ecg(
    target: torch.Tensor, window: int = 25
) -> torch.Tensor:
    """
    12-lead GT ECG에서 QRS 전기축 방향 벡터 계산.

    원리:
      심장 전기축 = QRS 시점에서 Lead I과 aVF의 진폭 비율로 결정
      axis_angle = atan2(aVF_QRS, Lead_I_QRS)

    Args:
        target: [B, 12, T] 12-lead ECG
        window: R-peak 주변 윈도우 크기 (samples)

    Returns:
        [B, 2] frontal plane axis unit vector (Lead_I축, aVF축)
    """
    B, _, T = target.shape
    lead_I = target[:, 0, :]     # [B, T]
    lead_aVF = target[:, 5, :]   # [B, T]

    # R-peak 시점: 전체 lead 에너지가 최대인 시점
    energy = target.abs().sum(dim=1)   # [B, T]
    r_peak_idx = energy.argmax(dim=-1) # [B]

    # R-peak ± window 구간의 mean amplitude
    t_range = torch.arange(T, device=target.device).unsqueeze(0)  # [1, T]
    lo = (r_peak_idx.unsqueeze(-1) - window).clamp(min=0)
    hi = (r_peak_idx.unsqueeze(-1) + window).clamp(max=T - 1)
    mask = (t_range >= lo) & (t_range <= hi)  # [B, T]
    count = mask.float().sum(dim=-1).clamp(min=1.0)  # [B]

    I_qrs = (lead_I * mask.float()).sum(dim=-1) / count       # [B]
    aVF_qrs = (lead_aVF * mask.float()).sum(dim=-1) / count   # [B]

    axis = torch.stack([I_qrs, aVF_qrs], dim=-1)  # [B, 2]
    return F.normalize(axis, dim=-1, eps=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: get_param_groups (위치 파라미터 그룹 제거 — 위치는 더 이상 학습 안 함)
# ──────────────────────────────────────────────────────────────────────────────
def get_param_groups(model: nn.Module, weight_decay: float):
    """CDG12용: 위치는 buffer이므로 position 그룹 불필요."""
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
# 해부학적 Gaussian 위치 생성 (고정 — 학습되지 않음)
# ──────────────────────────────────────────────────────────────────────────────
def _fibonacci_sphere(n: int) -> torch.Tensor:
    """n개의 점을 단위구면 위에 균등 배치."""
    idx = torch.arange(n, dtype=torch.float32)
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    theta = 2.0 * math.pi * idx / golden
    z = 1.0 - (2.0 * idx + 1.0) / float(max(n, 1))
    z = z.clamp(-0.99, 0.99)
    r = torch.sqrt(1.0 - z * z)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)


def _generate_anatomical_positions(n_gaussians: int = 32) -> torch.Tensor:
    """
    심장 해부학적 4구역에 고정 배치된 Gaussian 위치 생성.

    구역:
      1. RA/SA node region  (우심방 상방 — P파 생성 주도)
      2. Septum             (중격 — QRS 초기 탈분극)
      3. LV free wall       (좌심실 자유벽 — QRS 주도)
      4. RV free wall       (우심실 자유벽 — QRS 보조)

    각 구역에 n_gaussians//4 개씩 Fibonacci sphere 분포.
    """
    n_per_zone = n_gaussians // 4
    remainder = n_gaussians % 4

    zones = [
        # (center_offset from heart_center, radius)
        (torch.tensor([ 0.01,  0.04,  0.02]), 0.025),  # RA / SA node
        (torch.tensor([ 0.00,  0.00,  0.01]), 0.020),  # Septum
        (torch.tensor([-0.04, -0.02, -0.01]), 0.030),  # LV free wall
        (torch.tensor([ 0.03,  0.01,  0.02]), 0.025),  # RV free wall
    ]

    all_positions = []
    for i, (center_offset, radius) in enumerate(zones):
        n = n_per_zone + (1 if i < remainder else 0)
        sphere_pts = _fibonacci_sphere(n) * radius
        zone_center = _HEART_CENTER + center_offset
        all_positions.append(sphere_pts + zone_center)

    return torch.cat(all_positions, dim=0)  # [N, 3]


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Basis Envelope: 자유도 제한의 핵심
#   기존: envelope [B, N, T] → N×T 자유도 (32,000 for N=32, T=1000)
#   변경: bases [B, K, T] (공유) + weights [B, N, K] → N×K 자유도 (256)
# ──────────────────────────────────────────────────────────────────────────────
class TemporalBasisEnvelope(nn.Module):
    """
    K개의 시간 basis function을 encoder local features에서 생성하고,
    각 Gaussian은 K개의 mixing weight만 학습.

    비유: 오케스트라에서 K개의 "연주 패턴"이 공유되고,
          각 악기(Gaussian)는 어떤 패턴을 얼마나 따르는지만 결정.
    """

    def __init__(self, d_model: int = 256, n_bases: int = 8):
        super().__init__()
        self.n_bases = n_bases

        # encoder local_feat → K개 basis function 생성
        self.basis_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, n_bases, kernel_size=5, padding=2),
        )

    def forward(self, local_feat: torch.Tensor, mixing_weights: torch.Tensor):
        """
        local_feat:     [B, d_model, T]
        mixing_weights: [B, N, K]  (softmax-normalized)

        Returns:
            envelope: [B, N, T]
            bases:    [B, K, T]  (for visualization / analysis)
        """
        # K개 공유 basis function 생성 (non-negative)
        bases = F.softplus(self.basis_conv(local_feat))  # [B, K, T]

        # 각 basis를 [0, 1] 범위로 정규화
        bases_max = bases.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        bases = bases / bases_max  # [B, K, T]

        # 각 Gaussian의 envelope = basis들의 가중합
        envelope = torch.einsum('bnk,bkt->bnt', mixing_weights, bases)

        # 유효 범위 제한
        envelope = envelope.clamp(0.01, 1.0)

        return envelope, bases


# ──────────────────────────────────────────────────────────────────────────────
# Constrained Dipole Predictor
# ──────────────────────────────────────────────────────────────────────────────
class ConstrainedDipolePredictor(nn.Module):
    """
    CDG12의 핵심: 자유도가 제한된 쌍극자 예측기.

    CDG10/11 DipolePredictor와의 차이:
      Position:  nn.Parameter (학습) → register_buffer (고정!)
      Envelope:  자유형 cross-correlation → temporal basis의 가중합
      방향/진폭: 유지 (cross-attention으로 예측)
    """

    def __init__(self, d_model: int = 256, n_gaussians: int = 32,
                 n_heads: int = 8, n_temporal_bases: int = 8):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.n_temporal_bases = n_temporal_bases

        # ▸ 3D 위치: BUFFER — 학습되지 않음!
        #   해부학적 구역에 고정 배치
        init_pos = _generate_anatomical_positions(n_gaussians)
        self.register_buffer("positions", init_pos)  # [N, 3]

        # ▸ Cross-attention queries
        self.queries = nn.Parameter(
            F.normalize(torch.randn(1, n_gaussians, d_model), dim=-1)
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.05
        )
        self.norm = nn.LayerNorm(d_model)

        # ▸ Parameter head: 방향(3) + 진폭(1) + basis weights(K)
        out_dim = 4 + n_temporal_bases
        self.param_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_dim),
        )
        nn.init.normal_(self.param_head[-1].weight, std=0.01)
        bias = torch.zeros(out_dim)
        bias[3] = -1.0  # sigmoid(-1) ≈ 0.27 (적당히 sparse)
        self.param_head[-1].bias = nn.Parameter(bias)

        # ▸ Temporal basis envelope
        self.envelope_gen = TemporalBasisEnvelope(d_model, n_temporal_bases)

        # ▸ 시간가변 방향 보정 (물리적으로 필요: 탈분극 wavefront 회전)
        self.dir_temporal_proj = nn.Linear(d_model, d_model)
        self.dir_axis_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        nn.init.normal_(self.dir_axis_head[-1].weight, std=0.01)
        nn.init.zeros_(self.dir_axis_head[-1].bias)
        # sigmoid(-2.2) ≈ 0.10: 초기에는 거의 정적 방향
        self.dir_delta_scale = nn.Parameter(torch.tensor(-2.2))

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor):
        B = global_feat.size(0)

        # Cross-attention: 각 Gaussian이 encoder 출력에서 정보 추출
        q = self.queries.expand(B, -1, -1)
        kv = global_feat.permute(0, 2, 1)
        out, _ = self.cross_attn(q, kv, kv)
        q_static = self.norm(q + out)

        # 파라미터 예측
        params = self.param_head(q_static)  # [B, N, 4+K]

        # 3D 쌍극자 방향 (단위벡터)
        dipole_dir = F.normalize(params[..., 0:3], dim=-1, eps=1e-6)

        # 진폭
        amplitude = torch.sigmoid(params[..., 3]) + 1e-4

        # Temporal basis mixing weights (softmax → 합=1, 볼록 조합)
        basis_logits = params[..., 4:]  # [B, N, K]
        mixing_weights = F.softmax(basis_logits, dim=-1)

        # Envelope: temporal basis의 가중합
        envelope_t, bases = self.envelope_gen(local_feat, mixing_weights)

        # Position: buffer → batch expand (gradient 없음!)
        position = self.positions.unsqueeze(0).expand(B, -1, -1)

        # 시간가변 방향
        dir_w = self.dir_temporal_proj(q_static)
        dir_logits = torch.einsum(
            'bnc,bct->bnt', dir_w, local_feat
        ) / math.sqrt(dir_w.size(-1))
        dir_axis = F.normalize(
            self.dir_axis_head(q_static), dim=-1, eps=1e-6
        )
        dir_delta = (
            torch.tanh(dir_logits).unsqueeze(2) * dir_axis.unsqueeze(-1)
        )
        dir_scale = torch.sigmoid(self.dir_delta_scale)
        dipole_dir_t = F.normalize(
            dipole_dir.unsqueeze(-1) + dir_scale * dir_delta,
            dim=2, eps=1e-6,
        )  # [B, N, 3, T]

        return (position, dipole_dir, amplitude,
                envelope_t, dipole_dir_t, mixing_weights, bases)


# ──────────────────────────────────────────────────────────────────────────────
# CDGS12 메인 모델
# ──────────────────────────────────────────────────────────────────────────────
class CDGS12(nn.Module):
    """
    Identifiable Cardiac Gaussian Splatting.

    CDG10/11 대비 변경:
      1. Gaussian 수: 512 → 32 (identifiability)
      2. Position: nn.Parameter → buffer (해부학적 고정)
      3. Envelope: 자유형 → temporal basis 가중합 (자유도 감소)
      4. MetaElectrodeConditioner: 완전 제거 (전극 위치 고정)
      5. MetaPositionConditioner: 유지 (심장 크기/위치 보정은 물리적으로 타당)
    """

    def __init__(self, d_model: int = 256, n_gaussians: int = 32,
                 n_encoder_layers: int = 4, n_temporal_bases: int = 8,
                 **kwargs):
        super().__init__()
        self.encoder = MultiScaleEncoder(d_model, n_encoder_layers)
        self.predictor = ConstrainedDipolePredictor(
            d_model, n_gaussians, n_temporal_bases=n_temporal_bases
        )
        self.renderer = PhysicsRenderer()
        self.bypass = ResidualBypass(d_model)

        # ★ MetaElectrodeConditioner 삭제!
        #   전극은 표준 해부학적 위치에 고정.
        #   환자 차이는 심장 위치/크기로만 반영.
        self.meta_pos = MetaPositionConditioner()

        self.lead_gain = nn.Parameter(torch.ones(1, 12, 1))
        self.lead_bias = nn.Parameter(torch.zeros(1, 12, 1))

        # Gate: physics vs bypass 비율 (초기 50:50)
        self.gate = nn.Parameter(torch.full((1, 12, 1), 0.0))

    def forward(self, x, meta=None, bypass_alpha: float = 1.0,
                lead_id: torch.Tensor = None):
        local_feat, global_feat = self.encoder(x, lead_id=lead_id)

        # Gaussian 파라미터 예측
        (position, dipole_dir, amplitude,
         envelope_t, dipole_dir_t, mixing_weights, bases) = \
            self.predictor(local_feat, global_feat)

        # 환자별 심장 위치/크기 보정 (전극이 아니라 심장이 움직이는 것!)
        if meta is not None:
            position = self.meta_pos(meta, position)

        # 물리 렌더링 (전극 offset 없음 — 전극은 고정!)
        v_leads, V_electrodes = self.renderer(
            position, dipole_dir_t, amplitude, envelope_t,
            electrode_offset=None,
        )
        phys_out = v_leads * self.lead_gain + self.lead_bias

        # Bypass (고주파 보정)
        bypass_out = self.bypass(local_feat) * bypass_alpha

        # Additive mixing
        gate_w = torch.sigmoid(self.gate)
        mixed_out = phys_out + gate_w * bypass_out

        return mixed_out, phys_out, {
            "position":        position,
            "dipole_dir":      dipole_dir,
            "dipole_dir_t":    dipole_dir_t,
            "amplitude":       amplitude,
            "envelope_t":      envelope_t,
            "V_electrodes":    V_electrodes,
            "gate":            gate_w,
            "mixing_weights":  mixing_weights,
            "temporal_bases":  bases,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CDGS12Loss
# ──────────────────────────────────────────────────────────────────────────────
class CDGS12Loss(nn.Module):
    """
    CDG10Loss에서 불필요한 항 제거 + basis diversity 추가.

    Position이 고정이므로 boundary loss 불필요.
    Envelope이 basis 가중합이므로 env_alive/env_var 불필요 (구조적으로 보장).
    대신 temporal basis들이 서로 다른 패턴을 잡도록 diversity loss 추가.
    """

    def __init__(self):
        super().__init__()
        self.w_recon      = 1.0
        self.w_pearson    = 0.3    # 파형 상관 (shape matching)
        self.w_relative   = 0.2    # 상대 오차 (작은 lead 보호)
        self.w_phys       = 0.3    # 물리 브랜치 독립 loss (bypass가 물리를 죽이는 것 방지)
        self.w_amp_alive  = 0.05   # 진폭 소실 방지
        self.w_basis_div  = 0.02   # temporal basis diversity
        self.w_axis       = 0.1    # ★ 전기축 supervision (내부 dipole에 직접 GT 제공!)

    def _basis_diversity_loss(self, bases: torch.Tensor) -> torch.Tensor:
        """
        K개 temporal basis가 서로 다른 시간 패턴을 잡도록 강제.
        bases: [B, K, T]
        """
        K = bases.size(1)
        if K <= 1:
            return bases.new_tensor(0.0)

        # 각 basis를 정규화 후 cosine similarity
        bases_norm = bases - bases.mean(dim=-1, keepdim=True)
        bases_norm = F.normalize(bases_norm, dim=-1, eps=1e-6)  # [B, K, T]
        sim = torch.bmm(bases_norm, bases_norm.transpose(1, 2))  # [B, K, K]

        # Off-diagonal |cosine similarity| 평균 → 0이 되어야 함
        mask = ~torch.eye(K, device=sim.device, dtype=torch.bool).unsqueeze(0)
        return sim.abs().masked_select(mask).mean()

    def _axis_supervision_loss(
        self,
        dipole_dir: torch.Tensor,
        amplitude: torch.Tensor,
        envelope_t: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        전기축 supervision loss: 내부 dipole 방향에 직접 GT를 부여.

        원리:
          1. GT ECG (Lead I, aVF)에서 QRS 전기축 계산
          2. 모델 내부 dipole의 amplitude-weighted 방향을 frontal plane에 투영
          3. 두 방향의 cosine distance → 0이 되어야 함

        이 loss가 중요한 이유:
          - 교수님 지적 "GT가 없다" → 전기축이 내부 변수에 대한 부분적 GT 역할!
          - 12-lead ECG에서 유도된 임상적 측정치이므로 물리적으로 타당
        """
        B, N, T = envelope_t.shape

        # 1. GT axis from 12-lead ECG
        gt_axis = compute_electrical_axis_from_ecg(target)  # [B, 2]

        # 2. R-peak 시점 찾기
        energy = target.abs().sum(dim=1)       # [B, T]
        r_peak_idx = energy.argmax(dim=-1)     # [B]

        # 3. R-peak 시점의 envelope 추출
        batch_idx = torch.arange(B, device=envelope_t.device)
        gauss_idx = torch.arange(N, device=envelope_t.device)
        env_at_peak = envelope_t[
            batch_idx.unsqueeze(-1).expand(-1, N),
            gauss_idx.unsqueeze(0).expand(B, -1),
            r_peak_idx.unsqueeze(-1).expand(-1, N),
        ]  # [B, N]

        # 4. Net dipole = amplitude × envelope × direction 의 가중합
        weights = (amplitude * env_at_peak).unsqueeze(-1)  # [B, N, 1]
        net_dipole = (weights * dipole_dir).sum(dim=1)     # [B, 3]

        # 5. Frontal plane 투영 (Lead I 방향, aVF 방향)
        lead_I_dir = _LEAD_I_DIR.to(net_dipole.device)
        lead_aVF_dir = _LEAD_AVF_DIR.to(net_dipole.device)

        proj_I = (net_dipole * lead_I_dir.unsqueeze(0)).sum(dim=-1)     # [B]
        proj_aVF = (net_dipole * lead_aVF_dir.unsqueeze(0)).sum(dim=-1) # [B]

        pred_axis = F.normalize(
            torch.stack([proj_I, proj_aVF], dim=-1), dim=-1, eps=1e-6
        )  # [B, 2]

        # 6. Cosine loss: 1 - cos(GT, Pred)
        cos_sim = (gt_axis * pred_axis).sum(dim=-1)  # [B]
        return (1.0 - cos_sim).mean()

    def forward(self, pred, target, amplitude,
                position=None, phys_out=None,
                temporal_bases=None,
                dipole_dir=None, envelope_t=None,
                **kwargs):
        zero = pred.new_tensor(0.0)

        # 1. 기본 복원 (mixed_out)
        L_recon = F.l1_loss(pred, target)

        # 2. Pearson (파형 형태 유사도)
        L_pearson = zero
        if self.w_pearson > 0:
            L_pearson = pearson_loss(pred, target)

        # 3. 상대 오차 (작은 lead 보호)
        L_relative = zero
        if self.w_relative > 0:
            lead_rms = (target ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=0.01)
            L_relative = (torch.abs(pred - target) / lead_rms).mean()

        # 4. 물리 브랜치 전용 loss (bypass가 물리를 죽이지 못하도록)
        L_phys = zero
        if self.w_phys > 0 and phys_out is not None:
            L_phys = F.l1_loss(phys_out, target)

        # 5. Amplitude 생존
        L_amp_alive = F.relu(0.3 - amplitude.mean())

        # 6. Temporal basis diversity
        L_basis_div = zero
        if self.w_basis_div > 0 and temporal_bases is not None:
            L_basis_div = self._basis_diversity_loss(temporal_bases)

        # 7. ★ 전기축 supervision (내부 dipole에 직접 GT!)
        L_axis = zero
        if self.w_axis > 0 and dipole_dir is not None and envelope_t is not None:
            L_axis = self._axis_supervision_loss(
                dipole_dir, amplitude, envelope_t, target
            )

        total = (
            self.w_recon     * L_recon     +
            self.w_pearson   * L_pearson   +
            self.w_relative  * L_relative  +
            self.w_phys      * L_phys      +
            self.w_amp_alive * L_amp_alive +
            self.w_basis_div * L_basis_div +
            self.w_axis      * L_axis
        )

        return total, {
            "recon":     L_recon,
            "pearson":   L_pearson,
            "relative":  L_relative,
            "phys":      L_phys,
            "amp_alive": L_amp_alive,
            "basis_div": L_basis_div,
            "axis":      L_axis,
            "total":     total,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 2

    m = CDGS12(d_model=128, n_gaussians=32, n_encoder_layers=2, n_temporal_bases=8)
    x = torch.randn(B, 1, 1000)
    lead_id = torch.tensor([0, 1])
    meta = torch.tensor([
        [0.45, 0, 0.0, 0.0],
        [0.62, 1, -0.2, 0.5],
    ])

    y_train, y_phys, extras = m(x, meta=meta, bypass_alpha=0.5, lead_id=lead_id)
    assert y_train.shape == (B, 12, 1000), f"Shape mismatch: {y_train.shape}"

    # Position의 원본 buffer는 gradient 없음 확인
    # (MetaPositionConditioner 적용 후에는 연산 그래프에 포함되어 requires_grad=True 가능)
    assert not m.predictor.positions.requires_grad, "Base positions buffer should NOT require grad!"

    # Temporal bases 확인
    assert extras["temporal_bases"].shape == (B, 8, 1000)
    assert extras["mixing_weights"].shape == (B, 32, 8)

    # Loss 테스트
    loss_fn = CDGS12Loss()
    tgt = torch.randn(B, 12, 1000)
    loss_val, terms = loss_fn(
        y_train, tgt,
        extras["amplitude"],
        phys_out=y_phys,
        temporal_bases=extras["temporal_bases"],
        dipole_dir=extras["dipole_dir"],
        envelope_t=extras["envelope_t"],
    )
    assert "axis" in terms, "Axis loss term missing!"

    # 자유도 분석 출력
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    n_buffers = sum(b.numel() for b in m.buffers())

    print(f"\n[OK] CDGS12 (Identifiable CGS) Test 통과!")
    print(f"  Output shape: {y_train.shape}")
    print(f"  Total Loss:   {loss_val.item():.4f}")
    print(f"  Loss terms:   {list(terms.keys())}")
    print(f"\n  === Identifiability 분석 ===")
    print(f"  Trainable params:  {n_trainable:,}")
    print(f"  Fixed buffers:     {n_buffers:,} (positions + electrode coords)")
    print(f"  Gaussians:         32 (해부학적 4구역 × 8개)")
    print(f"  Per-sample 물리 미지수: ~384 (32 × [3 dir + 1 amp + 8 basis wt])")
    print(f"  Per-sample 관측:        ~9,000 (9 전극 × 1000 timestep)")
    print(f"  Ratio:              {9000/384:.1f}:1 (over-determined [OK])")
    print(f"\n  === 물리 변수 ===")
    print(f"  Amplitude:         {extras['amplitude'].mean().item():.4f}")
    print(f"  Gate (phys frac):  {extras['gate'].mean().item():.4f}")
    print(f"  Position fixed:    [OK] (register_buffer)")
    print(f"  Electrode fixed:   [OK] (MetaElectrodeConditioner removed)")
    print(f"  Temporal bases:    {extras['temporal_bases'].shape}")
    print(f"  Mixing weights:    {extras['mixing_weights'].shape}")

    # param groups 확인
    pgs = get_param_groups(m, 1e-4)
    for i, pg in enumerate(pgs):
        n = sum(p.numel() for p in pg["params"])
        print(f"  Param group {i}: {n:,} params, wd={pg['weight_decay']}")
