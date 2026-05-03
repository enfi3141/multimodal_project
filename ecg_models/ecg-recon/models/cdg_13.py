# -*- coding: utf-8 -*-
"""
CDGS v13 (`cdg_13.py`): Two-Stage Cardiac Gaussian Splatting

핵심 아이디어:
  Stage 1 (Pretrain): 12-lead → 3D Gaussian → 12-lead (Autoencoder)
    - 12-lead 전체를 보고 3D 가우시안 표현을 학습
    - 정보가 충분하므로 물리 모델이 의미 있는 표현을 배울 수 있음
    - 가우시안 위치(Position)도 학습 가능 (nn.Parameter)

  Stage 2 (Finetune): 1-lead → 학습된 3D Gaussian → 12-lead
    - Stage 1의 물리 렌더러 + 가우시안 위치를 고정(freeze)
    - 새로운 1-lead 인코더가 동일한 가우시안 파라미터 공간으로 매핑
    - 증류(Distillation) Loss: 12-lead 인코더의 출력을 모방

비전 3DGS와의 완벽한 대응:
  3DGS Stage 1: 100장 사진 → 3D 씬 학습
  3DGS Stage 2: 1장 사진 → 새로운 시점 생성
  CDG13 Stage 1: 12-lead → 3D 심장 전기 학습
  CDG13 Stage 2: 1-lead → 나머지 11-lead 생성
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CDG12에서 공유 컴포넌트 임포트 ──
try:
    from .cdg_12 import (
        _ELECTRODE_POSITIONS, _HEART_CENTER,
        MetaPositionConditioner,
        DilatedConvBlock, BeatAlignedStem, MultiScaleEncoder,
        PhysicsRenderer, LearnedAttenuation, ResidualBypass,
        pearson_loss, TemporalBasisEnvelope,
        compute_electrical_axis_from_ecg,
        _LEAD_I_DIR, _LEAD_AVF_DIR,
    )
except ImportError:
    from cdg_12 import (
        _ELECTRODE_POSITIONS, _HEART_CENTER,
        MetaPositionConditioner,
        DilatedConvBlock, BeatAlignedStem, MultiScaleEncoder,
        PhysicsRenderer, LearnedAttenuation, ResidualBypass,
        pearson_loss, TemporalBasisEnvelope,
        compute_electrical_axis_from_ecg,
        _LEAD_I_DIR, _LEAD_AVF_DIR,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fibonacci sphere (유연한 개수의 가우시안 초기 위치 생성용)
# ──────────────────────────────────────────────────────────────────────────────
def _fibonacci_sphere(n: int) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.float32)
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    theta = 2.0 * math.pi * idx / golden
    z = 1.0 - (2.0 * idx + 1.0) / float(max(n, 1))
    z = z.clamp(-0.99, 0.99)
    r = torch.sqrt(1.0 - z * z)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)


def _generate_anatomical_positions_flex(n_gaussians: int = 64) -> torch.Tensor:
    """
    AHA 기반 5개 해부학적 구역에 비율대로 가우시안 분배.

    구역 비율:
      LV free wall  35%  |  RV free wall  20%  |  Septum  20%
      RA (우심방)   12.5% |  LA (좌심방)  12.5%
    """
    zones = [
        (torch.tensor([-0.02, -0.02, -0.01]), 0.035, 0.35),  # LV
        (torch.tensor([ 0.03,  0.01,  0.02]), 0.030, 0.20),  # RV
        (torch.tensor([ 0.00,  0.00,  0.01]), 0.025, 0.20),  # Septum
        (torch.tensor([ 0.02,  0.03,  0.04]), 0.020, 0.125), # RA
        (torch.tensor([-0.02, -0.03,  0.04]), 0.020, 0.125), # LA
    ]
    counts = [max(1, round(n_gaussians * p)) for _, _, p in zones]
    counts[0] += n_gaussians - sum(counts)

    all_pos = []
    for (offset, radius, _), count in zip(zones, counts):
        pts = _fibonacci_sphere(count) * radius
        all_pos.append(pts + _HEART_CENTER + offset)
    return torch.cat(all_pos, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# 12-lead 인코더 (Stage 1 전용)
# ──────────────────────────────────────────────────────────────────────────────
class BeatAlignedStem12(nn.Module):
    """12채널 입력용 스템."""
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.stem = nn.Conv1d(12, d_model, kernel_size=7, padding=3)
        self.bn   = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.bn(self.stem(x)))


class MultiScaleEncoder12(nn.Module):
    """12-lead 전용 인코더 (Stage 1)."""
    def __init__(self, d_model: int = 256, n_layers: int = 4):
        super().__init__()
        self.stem       = BeatAlignedStem12(d_model)
        self.cnn_blocks = nn.ModuleList([DilatedConvBlock(d_model, d_model) for _ in range(3)])
        self.cnn_norms  = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(3)])
        self.downsample = nn.Conv1d(d_model, d_model, kernel_size=4, stride=4, padding=0)
        self.tf_layers  = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
                batch_first=True, dropout=0.1
            ) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor):
        h = self.stem(x)
        for block, norm in zip(self.cnn_blocks, self.cnn_norms):
            h = h + norm(block(h))
        local_feat = h
        h_seq = self.downsample(h).permute(0, 2, 1)
        for layer in self.tf_layers:
            h_seq = layer(h_seq)
        return local_feat, h_seq.permute(0, 2, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 가우시안 파라미터 예측기 (Stage 1/2 공유)
# ──────────────────────────────────────────────────────────────────────────────
class GaussianPredictor(nn.Module):
    """
    CDG12의 ConstrainedDipolePredictor와 차이:
      - Position: register_buffer → nn.Parameter (Stage 1에서 학습!)
      - n_gaussians: 32 → 64 (더 풍부한 표현력)
      - n_temporal_bases: 8 → 16
    """
    def __init__(self, d_model: int = 256, n_gaussians: int = 64,
                 n_heads: int = 8, n_temporal_bases: int = 16):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.n_temporal_bases = n_temporal_bases

        # ▸ 위치: nn.Parameter (Stage 1 학습 → Stage 2 고정)
        init_pos = _generate_anatomical_positions_flex(n_gaussians)
        self.positions = nn.Parameter(init_pos)  # [N, 3]

        # ▸ Cross-attention
        self.queries = nn.Parameter(
            F.normalize(torch.randn(1, n_gaussians, d_model), dim=-1)
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.05
        )
        self.norm = nn.LayerNorm(d_model)

        # ▸ Parameter head: dir(3) + amp(1) + basis_weights(K)
        out_dim = 4 + n_temporal_bases
        self.param_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, out_dim),
        )
        nn.init.normal_(self.param_head[-1].weight, std=0.01)
        bias = torch.zeros(out_dim); bias[3] = -1.0
        self.param_head[-1].bias = nn.Parameter(bias)

        # ▸ Temporal basis
        self.envelope_gen = TemporalBasisEnvelope(d_model, n_temporal_bases)

        # ▸ Time-varying direction
        self.dir_temporal_proj = nn.Linear(d_model, d_model)
        self.dir_axis_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        nn.init.normal_(self.dir_axis_head[-1].weight, std=0.01)
        nn.init.zeros_(self.dir_axis_head[-1].bias)
        self.dir_delta_scale = nn.Parameter(torch.tensor(-2.2))

    def forward(self, local_feat, global_feat):
        B = global_feat.size(0)
        q = self.queries.expand(B, -1, -1)
        kv = global_feat.permute(0, 2, 1)
        out, _ = self.cross_attn(q, kv, kv)
        q_static = self.norm(q + out)

        params = self.param_head(q_static)
        dipole_dir = F.normalize(params[..., 0:3], dim=-1, eps=1e-6)
        amplitude  = torch.sigmoid(params[..., 3]) + 1e-4
        mixing_w   = F.softmax(params[..., 4:], dim=-1)
        envelope_t, bases = self.envelope_gen(local_feat, mixing_w)
        position = self.positions.unsqueeze(0).expand(B, -1, -1)

        # Time-varying direction
        dir_w = self.dir_temporal_proj(q_static)
        dir_logits = torch.einsum('bnc,bct->bnt', dir_w, local_feat) / math.sqrt(dir_w.size(-1))
        dir_axis = F.normalize(self.dir_axis_head(q_static), dim=-1, eps=1e-6)
        dir_delta = torch.tanh(dir_logits).unsqueeze(2) * dir_axis.unsqueeze(-1)
        dir_scale = torch.sigmoid(self.dir_delta_scale)
        dipole_dir_t = F.normalize(
            dipole_dir.unsqueeze(-1) + dir_scale * dir_delta, dim=2, eps=1e-6,
        )
        return (position, dipole_dir, amplitude,
                envelope_t, dipole_dir_t, mixing_w, bases)


# ──────────────────────────────────────────────────────────────────────────────
# CDGS13 메인 모델
# ──────────────────────────────────────────────────────────────────────────────
class CDGS13(nn.Module):
    """
    Two-Stage Cardiac Gaussian Splatting.

    Stage 1: 12-lead → Encoder12 → GaussianPredictor → PhysicsRenderer → 12-lead
    Stage 2: 1-lead  → Encoder1  → GaussianPredictor → PhysicsRenderer → 12-lead
                                   (positions frozen)   (renderer frozen)
    """
    def __init__(self, d_model: int = 256, n_gaussians: int = 64,
                 n_encoder_layers: int = 4, n_temporal_bases: int = 16,
                 **kwargs):
        super().__init__()
        # Stage 1: 12-lead encoder
        self.encoder_12 = MultiScaleEncoder12(d_model, n_encoder_layers)
        # Stage 2: 1-lead encoder
        self.encoder_1  = MultiScaleEncoder(d_model, n_encoder_layers)

        # 공유: 가우시안 예측기 + 물리 렌더러
        self.predictor = GaussianPredictor(
            d_model, n_gaussians, n_temporal_bases=n_temporal_bases
        )
        self.renderer  = PhysicsRenderer()
        self.meta_pos  = MetaPositionConditioner()

        self.lead_gain = nn.Parameter(torch.ones(1, 12, 1))
        self.lead_bias = nn.Parameter(torch.zeros(1, 12, 1))

        # Stage별 bypass
        self.bypass_12 = ResidualBypass(d_model)
        self.bypass_1  = ResidualBypass(d_model)
        self.gate      = nn.Parameter(torch.full((1, 12, 1), 0.0))

        self._stage = 1

    @property
    def stage(self):
        return self._stage

    def set_stage(self, stage: int):
        """Stage 전환. Stage 2 진입 시 Stage 1 컴포넌트를 고정."""
        assert stage in (1, 2)
        self._stage = stage
        if stage == 2:
            # Stage 1에서 배운 것들을 고정
            self.encoder_12.requires_grad_(False)
            self.encoder_12.eval()
            self.predictor.positions.requires_grad_(False)
            self.renderer.requires_grad_(False)
            self.lead_gain.requires_grad_(False)
            self.lead_bias.requires_grad_(False)
            self.bypass_12.requires_grad_(False)

            # ★ 증류용 predictor 스냅샷 (Stage 1 학습 결과 고정 복사)
            #   동일한 predictor로 teacher/student를 동시에 돌리면
            #   teacher target이 매 step 변동 → NaN 발산
            self._predictor_frozen = copy.deepcopy(self.predictor)
            self._predictor_frozen.requires_grad_(False)
            self._predictor_frozen.eval()

            print("[CDG13] Stage 2 activated:")
            print("  Frozen: encoder_12, positions, renderer, lead_gain/bias, bypass_12")
            print("  Frozen: predictor snapshot for distillation")
            print("  Trainable: encoder_1, predictor (except pos), bypass_1, gate")

    def forward(self, x, meta=None, bypass_alpha: float = 1.0,
                lead_id: torch.Tensor = None, x_12: torch.Tensor = None):
        """
        Args:
            x:     Stage 1 → [B, 12, T],  Stage 2 → [B, 1, T]
            x_12:  Stage 2에서 증류용 12-lead GT (optional)
            meta:  [B, 4+] patient metadata
        """
        if self._stage == 1:
            local_feat, global_feat = self.encoder_12(x)
            bypass_net = self.bypass_12
        else:
            local_feat, global_feat = self.encoder_1(x, lead_id=lead_id)
            bypass_net = self.bypass_1

        (position, dipole_dir, amplitude,
         envelope_t, dipole_dir_t, mixing_w, bases) = \
            self.predictor(local_feat, global_feat)

        if meta is not None:
            position = self.meta_pos(meta, position)

        v_leads, V_electrodes = self.renderer(
            position, dipole_dir_t, amplitude, envelope_t,
            electrode_offset=None,
        )
        phys_out = v_leads * self.lead_gain + self.lead_bias
        # NaN 방지: physics output clamp
        phys_out = phys_out.clamp(-20.0, 20.0)

        bypass_out = bypass_net(local_feat) * bypass_alpha
        gate_w = torch.sigmoid(self.gate)
        mixed_out = phys_out + gate_w * bypass_out

        extras = {
            "position": position, "dipole_dir": dipole_dir,
            "dipole_dir_t": dipole_dir_t, "amplitude": amplitude,
            "envelope_t": envelope_t, "V_electrodes": V_electrodes,
            "gate": gate_w, "mixing_weights": mixing_w,
            "temporal_bases": bases,
        }

        # Stage 2 증류: frozen predictor 스냅샷으로 안정적인 teacher target 생성
        if self._stage == 2 and x_12 is not None and hasattr(self, '_predictor_frozen'):
            with torch.no_grad():
                self.encoder_12.eval()
                self._predictor_frozen.eval()
                loc12, glob12 = self.encoder_12(x_12)
                (_, dir12, amp12, env12, _, mix12, _) = \
                    self._predictor_frozen(loc12, glob12)
            extras["distill_dir"] = dir12
            extras["distill_amp"] = amp12
            extras["distill_env"] = env12
            extras["distill_mix"] = mix12

        return mixed_out, phys_out, extras


# ──────────────────────────────────────────────────────────────────────────────
# Param groups
# ──────────────────────────────────────────────────────────────────────────────
def get_param_groups(model: nn.Module, weight_decay: float):
    """CDG13용 파라미터 그룹. Stage에 따라 자동 필터링."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        elif "positions" in name:
            no_decay.append(param)  # positions는 weight_decay 없이
        else:
            decay.append(param)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# CDGS13Loss
# ──────────────────────────────────────────────────────────────────────────────
class CDGS13Loss(nn.Module):
    """
    Stage 1: 12-lead 오토인코더 reconstruction loss
    Stage 2: 12-lead reconstruction + 증류(distillation) loss
    """
    def __init__(self):
        super().__init__()
        # 공통
        self.w_recon    = 1.0
        self.w_pearson  = 0.3
        self.w_relative = 0.2
        self.w_phys     = 0.3
        self.w_amp_alive = 0.05
        self.w_basis_div = 0.02
        self.w_axis     = 0.1
        # Stage 2 전용
        self.w_distill  = 0.5

    def _basis_diversity_loss(self, bases: torch.Tensor) -> torch.Tensor:
        K = bases.size(1)
        if K <= 1:
            return bases.new_tensor(0.0)
        b_norm = F.normalize(bases - bases.mean(dim=-1, keepdim=True), dim=-1, eps=1e-6)
        sim = torch.bmm(b_norm, b_norm.transpose(1, 2))
        mask = ~torch.eye(K, device=sim.device, dtype=torch.bool).unsqueeze(0)
        return sim.abs().masked_select(mask).mean()

    def _axis_supervision_loss(self, dipole_dir, amplitude, envelope_t, target):
        B, N, T = envelope_t.shape
        gt_axis = compute_electrical_axis_from_ecg(target)
        energy = target.abs().sum(dim=1)
        r_peak_idx = energy.argmax(dim=-1)
        batch_idx = torch.arange(B, device=envelope_t.device)
        gauss_idx = torch.arange(N, device=envelope_t.device)
        env_at_peak = envelope_t[
            batch_idx.unsqueeze(-1).expand(-1, N),
            gauss_idx.unsqueeze(0).expand(B, -1),
            r_peak_idx.unsqueeze(-1).expand(-1, N),
        ]
        weights = (amplitude * env_at_peak).unsqueeze(-1)
        net_dipole = (weights * dipole_dir).sum(dim=1)
        lead_I_dir = _LEAD_I_DIR.to(net_dipole.device)
        lead_aVF_dir = _LEAD_AVF_DIR.to(net_dipole.device)
        proj_I   = (net_dipole * lead_I_dir.unsqueeze(0)).sum(dim=-1)
        proj_aVF = (net_dipole * lead_aVF_dir.unsqueeze(0)).sum(dim=-1)
        pred_axis = F.normalize(torch.stack([proj_I, proj_aVF], dim=-1), dim=-1, eps=1e-6)
        cos_sim = (gt_axis * pred_axis).sum(dim=-1)
        return (1.0 - cos_sim).mean()

    def _distillation_loss(self, extras):
        """Stage 2: 1-lead 인코더 출력이 12-lead 인코더 출력을 모방하도록."""
        loss = extras["amplitude"].new_tensor(0.0)
        if "distill_dir" not in extras:
            return loss
        # 방향 cosine distance
        cos = (extras["dipole_dir"] * extras["distill_dir"]).sum(dim=-1)
        loss = loss + (1.0 - cos).mean()
        # 진폭 MSE
        loss = loss + F.mse_loss(extras["amplitude"], extras["distill_amp"])
        # Mixing weights KL divergence
        log_p = torch.log(extras["mixing_weights"] + 1e-8)
        loss = loss + F.kl_div(log_p, extras["distill_mix"], reduction="batchmean")
        return loss

    def forward(self, pred, target, amplitude, stage=1,
                position=None, phys_out=None,
                temporal_bases=None, dipole_dir=None,
                envelope_t=None, extras=None, **kwargs):
        zero = pred.new_tensor(0.0)

        L_recon    = F.l1_loss(pred, target)
        L_pearson  = pearson_loss(pred, target) if self.w_pearson > 0 else zero
        L_relative = zero
        if self.w_relative > 0:
            rms = (target ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=0.01)
            L_relative = (torch.abs(pred - target) / rms).mean()
        L_phys = F.l1_loss(phys_out, target) if (self.w_phys > 0 and phys_out is not None) else zero
        L_amp  = F.relu(0.3 - amplitude.mean())
        L_bdiv = self._basis_diversity_loss(temporal_bases) if temporal_bases is not None else zero
        
        # 가우시안 간의 과동기화 방지 (Mode Collapse 해결)
        L_env_div = zero
        if envelope_t is not None:
            N = envelope_t.size(1)
            env_norm = envelope_t - envelope_t.mean(dim=-1, keepdim=True)
            env_norm = F.normalize(env_norm, dim=-1, eps=1e-6)
            sim = torch.bmm(env_norm, env_norm.transpose(1, 2))
            mask = ~torch.eye(N, device=sim.device, dtype=torch.bool).unsqueeze(0)
            L_env_div = sim.abs().masked_select(mask).mean()

        L_axis = zero
        if self.w_axis > 0 and dipole_dir is not None and envelope_t is not None:
            L_axis = self._axis_supervision_loss(dipole_dir, amplitude, envelope_t, target)

        # Stage 2 전용: 증류
        L_distill = zero
        if stage == 2 and extras is not None:
            L_distill = self._distillation_loss(extras)

        # NaN guard: 개별 항이 NaN이면 0으로 대체 (Stage 2 초기 불안정 방지)
        components = {
            "recon": L_recon, "pearson": L_pearson, "relative": L_relative,
            "phys": L_phys, "amp_alive": L_amp, "basis_div": L_bdiv,
            "env_div": L_env_div, "axis": L_axis, "distill": L_distill,
        }
        weights = {
            "recon": self.w_recon, "pearson": self.w_pearson,
            "relative": self.w_relative, "phys": self.w_phys,
            "amp_alive": self.w_amp_alive, "basis_div": self.w_basis_div,
            "env_div": 0.05, "axis": self.w_axis, "distill": self.w_distill,
        }
        total = zero
        for k, v in components.items():
            if torch.isnan(v):
                components[k] = zero  # NaN 항 제거
            else:
                total = total + weights[k] * v

        components["total"] = total
        return total, components


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 2

    m = CDGS13(d_model=128, n_gaussians=64, n_encoder_layers=2, n_temporal_bases=16)
    loss_fn = CDGS13Loss()
    meta = torch.tensor([[0.45, 0, 0.0, 0.0], [0.62, 1, -0.2, 0.5]])

    # ── Stage 1 테스트: 12-lead autoencoder ──
    print("=" * 60)
    print("Stage 1: 12-lead Autoencoder")
    print("=" * 60)
    x12 = torch.randn(B, 12, 1000)
    y1, phys1, ex1 = m(x12, meta=meta, bypass_alpha=0.5)
    assert y1.shape == (B, 12, 1000), f"Stage 1 shape mismatch: {y1.shape}"
    loss1, terms1 = loss_fn(
        y1, x12, ex1["amplitude"], stage=1,
        phys_out=phys1, temporal_bases=ex1["temporal_bases"],
        dipole_dir=ex1["dipole_dir"], envelope_t=ex1["envelope_t"],
    )
    print(f"  Output:    {y1.shape}")
    print(f"  Loss:      {loss1.item():.4f}")
    print(f"  Positions: {m.predictor.positions.shape} (requires_grad={m.predictor.positions.requires_grad})")
    print(f"  Gaussians: {m.predictor.n_gaussians}")

    # ── Stage 2 전환 ──
    print("\n" + "=" * 60)
    print("Stage 2: 1-lead → 12-lead")
    print("=" * 60)
    m.set_stage(2)

    x1 = torch.randn(B, 1, 1000)
    lead_id = torch.tensor([0, 0])
    y2, phys2, ex2 = m(x1, meta=meta, bypass_alpha=0.5,
                        lead_id=lead_id, x_12=x12)
    assert y2.shape == (B, 12, 1000), f"Stage 2 shape mismatch: {y2.shape}"
    assert not m.predictor.positions.requires_grad, "Positions should be frozen in Stage 2!"
    assert "distill_dir" in ex2, "Distillation targets missing!"

    tgt = torch.randn(B, 12, 1000)
    loss2, terms2 = loss_fn(
        y2, tgt, ex2["amplitude"], stage=2,
        phys_out=phys2, temporal_bases=ex2["temporal_bases"],
        dipole_dir=ex2["dipole_dir"], envelope_t=ex2["envelope_t"],
        extras=ex2,
    )
    print(f"  Output:    {y2.shape}")
    print(f"  Loss:      {loss2.item():.4f}")
    print(f"  Distill:   {terms2['distill'].item():.4f}")
    print(f"  Positions: frozen={not m.predictor.positions.requires_grad}")

    # ── 자유도 분석 ──
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_fixed = sum(b.numel() for b in m.buffers())
    N, K = m.predictor.n_gaussians, m.predictor.n_temporal_bases
    dof = N * (3 + 1 + K)  # dir + amp + basis weights
    obs = 9 * 1000
    print(f"\n  === Identifiability (Stage 2) ===")
    print(f"  Trainable:   {n_train:,}")
    print(f"  Gaussians:   {N}")
    print(f"  Temporal K:  {K}")
    print(f"  Per-sample DOF:  {dof}")
    print(f"  Per-sample Obs:  {obs}")
    print(f"  Ratio:           {obs/dof:.1f}:1 (over-determined [OK])")

    print(f"\n[OK] CDGS13 Two-Stage Test passed!")
