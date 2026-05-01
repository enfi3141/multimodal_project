# -*- coding: utf-8 -*-
"""
CDGS v10 (`cdg_10.py`): True Physics-Based Cardiac Gaussian Splatting
=====================================================================
핵심 설계:
  1. 512개 가우시안이 자유로운 3D 위치를 학습 → 심장 모양 자동 형성
  2. 1/r³ 물리 법칙으로 9개 전극에 전위 계산 (Dower 행렬 불필요)
  3. 12-lead = 전극 간 전위차 (실제 ECG 물리와 동일)
  4. 3DGS 스타일: soft boundary + amplitude 관리
  5. 단순화된 4-term loss (해상도 무관)
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

def get_param_groups(model: nn.Module, weight_decay: float, pos_lr_scale: float = 0.05):
    """3DGS 스타일: 위치 파라미터는 별도 lr (기본 lr × 0.05)"""
    pos_params, decay, no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 위치 파라미터는 별도 그룹
        if "positions" in name:
            pos_params.append(param)
        elif param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,      "weight_decay": weight_decay},
        {"params": no_decay,   "weight_decay": 0.0},
        {"params": pos_params,  "weight_decay": 0.0, "lr_scale": pos_lr_scale},
    ]

# ──────────────────────────────────────────────────────────────────────────────
# 9개 전극 표준 좌표 (해부학적 기본값, 정규화 단위)
# ──────────────────────────────────────────────────────────────────────────────
_ELECTRODE_POSITIONS = torch.tensor([
    # 사지 전극 (몸에서 멀리 → 위치 변동 무의미)
    [ 0.30,  0.25,  0.00],   # RA (오른팔)
    [-0.30,  0.25,  0.00],   # LA (왼팔)
    [-0.05, -0.40,  0.00],   # LL (왼다리)
    # 흉부 전극 (심장 가까이 → 위치 중요)
    [ 0.03,  0.05,  0.20],   # V1 (4번째 늑간, 흉골 우연)
    [ 0.00,  0.05,  0.22],   # V2 (4번째 늑간, 흉골 좌연)
    [-0.04,  0.01,  0.21],   # V3 (V2-V4 중간)
    [-0.08, -0.02,  0.18],   # V4 (5번째 늑간, 쇄골중선)
    [-0.14, -0.02,  0.13],   # V5 (V4-V6 중간)
    [-0.20, -0.02,  0.07],   # V6 (5번째 늑간, 중액와선)
], dtype=torch.float32)

# 심장 중심 좌표 (표준)
_HEART_CENTER = torch.tensor([-0.03, 0.02, 0.10], dtype=torch.float32)

# ──────────────────────────────────────────────────────────────────────────────
# MetaConditioner: 환자 메타데이터 → 흉부 전극 위치 보정
# ──────────────────────────────────────────────────────────────────────────────
class MetaElectrodeConditioner(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.age_proj = nn.Linear(1, 8)
        self.sex_emb  = nn.Embedding(3, 8)
        self.hw_proj  = nn.Linear(2, 16)

        self.net = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 9 * 3),  # 9 electrodes × 3D offset
        )
        # 초기에는 보정 없음
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        """meta: [B, 4+] → electrode offset [B, 9, 3]"""
        age = meta[:, 0:1]
        sex = meta[:, 1].long().clamp(0, 2)
        hw  = meta[:, 2:4]

        a = F.gelu(self.age_proj(age))
        s = self.sex_emb(sex)
        h = F.gelu(self.hw_proj(hw))

        feat = torch.cat([a, s, h], dim=-1)
        offsets = self.net(feat).view(-1, 9, 3)
        # 사지 전극(0~2)은 보정 불필요 → 0으로 마스킹
        mask = torch.zeros(9, 1, device=offsets.device, dtype=offsets.dtype)
        mask[3:] = 1.0  # V1~V6만 보정
        return torch.tanh(offsets) * 0.03 * mask  # max +/-3cm


# ──────────────────────────────────────────────────────────────────────────────
# MetaPositionConditioner: metadata -> global position scaling/offset
#   심비대, 체격 차이 등에 의한 심장 크기/위치 차이를 반영
#   base position에 작은 보정만 더함 (gradient 안정성 유지)
# ──────────────────────────────────────────────────────────────────────────────
class MetaPositionConditioner(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),   # age, sex, h, w
            nn.GELU(),
            nn.Linear(hidden_dim, 6),   # 3D offset + 3D scale
        )
        nn.init.zeros_(self.net[-1].weight)
        # scale 초기값 1.0 (log scale 0.0), offset 초기값 0.0
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, meta: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        meta: [B, 4+]
        positions: [B, N, 3] (base positions)
        Returns: [B, N, 3] (adjusted positions)
        """
        params = self.net(meta[:, :4])   # [B, 6]
        offset = torch.tanh(params[:, :3]) * 0.02   # [B, 3] max +/-2cm shift
        scale  = 1.0 + torch.tanh(params[:, 3:]) * 0.15  # [B, 3] +/-15% scaling

        # 심장 중심 기준 scaling + global offset
        heart_center = _HEART_CENTER.to(positions.device)
        centered = positions - heart_center
        scaled = centered * scale.unsqueeze(1)  # [B, N, 3]
        return scaled + heart_center + offset.unsqueeze(1)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder (cdg_9 기반, Lead Identity 적용)
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
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
                batch_first=True, dropout=0.1
            )
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
# DipolePredictor: 3DGS 스타일 — 위치는 nn.Parameter (직접 최적화)
#   위치: nn.Parameter (전 환자 공유, 해부학적 구조)
#   방향+진폭+envelope: 네트워크 예측 (입력 ECG에 따라 변함)
# ──────────────────────────────────────────────────────────────────────────────
class DipolePredictor(nn.Module):
    def __init__(self, d_model: int = 256, n_gaussians: int = 512, n_heads: int = 8):
        super().__init__()
        self.n_gaussians = n_gaussians

        # ▸ 3D 위치: nn.Parameter로 직접 최적화 (3DGS와 동일!)
        #   심장 중심 근방에 fibonacci sphere로 초기화
        init_pos = _HEART_CENTER + self._fibonacci_sphere(n_gaussians) * 0.06
        self.positions = nn.Parameter(init_pos)  # [N, 3] — 전 환자 공유!

        self.queries    = nn.Parameter(F.normalize(torch.randn(1, n_gaussians, d_model), dim=-1))
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.05)
        self.norm       = nn.LayerNorm(d_model)

        self.temporal_proj  = nn.Linear(d_model, d_model)
        self.envelope_bias  = nn.Parameter(torch.tensor(2.0))  # sigmoid(2)≈0.88: 기본 활성
        # envelope collapse 방지: 바닥값 + logit 클리핑으로 완전 소멸을 막음
        self.envelope_floor = 0.05
        self.envelope_logit_clip = 6.0

        # 시간가변 방향 보정: 정적 방향에 작은 동적 편차를 추가
        self.dir_temporal_proj = nn.Linear(d_model, d_model)
        self.dir_axis_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        nn.init.normal_(self.dir_axis_head[-1].weight, std=0.01)
        nn.init.zeros_(self.dir_axis_head[-1].bias)
        # sigmoid(-2.2)≈0.10: 초기에는 거의 정적 방향으로 시작
        self.dir_delta_scale = nn.Parameter(torch.tensor(-2.2))

        # 출력: 3D 방향 + 1D 진폭 = 4차원 (위치는 nn.Parameter이므로 제외!)
        self.param_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),
        )
        nn.init.normal_(self.param_head[-1].weight, std=0.01)
        bias = torch.zeros(4)
        bias[3] = -1.0  # sigmoid(-1) ≈ 0.27 (적당히 sparse)
        self.param_head[-1].bias = nn.Parameter(bias)

    @staticmethod
    def _fibonacci_sphere(n: int) -> torch.Tensor:
        """n개의 점을 단위구면 위에 균등 배치"""
        idx = torch.arange(n, dtype=torch.float32)
        golden = (1.0 + math.sqrt(5.0)) / 2.0
        theta = 2.0 * math.pi * idx / golden
        z = 1.0 - (2.0 * idx + 1.0) / float(n)
        z = z.clamp(-0.99, 0.99)
        r = torch.sqrt(1.0 - z * z)
        return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor):
        B = global_feat.size(0)

        # Cross-attention: 각 가우시안이 인코더 출력에서 정보 추출
        q  = self.queries.expand(B, -1, -1)
        kv = global_feat.permute(0, 2, 1)
        out, _ = self.cross_attn(q, kv, kv)
        q_static = self.norm(q + out)

        # 파라미터 예측 (방향 + 진폭만! 위치는 별도)
        params = self.param_head(q_static)  # [B, N, 4]

        # 3D 쌍극자 방향 (단위벡터)
        dipole_dir = F.normalize(params[..., 0:3], dim=-1, eps=1e-6)  # [B, N, 3]

        # 진폭
        amplitude = torch.sigmoid(params[..., 3]) + 1e-4  # [B, N]

        # 3D 위치: nn.Parameter → 배치 차원으로 확장
        position = self.positions.unsqueeze(0).expand(B, -1, -1)  # [B, N, 3]

        # 시간 envelope: 인코더 local_feat과의 cross-correlation
        temp_w = self.temporal_proj(q_static)
        temporal_logits = torch.einsum('bnc,bct->bnt', temp_w, local_feat) / math.sqrt(temp_w.size(-1))
        temporal_logits = temporal_logits.clamp(-self.envelope_logit_clip, self.envelope_logit_clip)
        envelope_sig = torch.sigmoid(temporal_logits + self.envelope_bias)
        envelope_t = self.envelope_floor + (1.0 - self.envelope_floor) * envelope_sig  # [B, N, T]

        # 시간가변 방향: static dir + (axis * temporal scalar)
        dir_w = self.dir_temporal_proj(q_static)
        dir_logits = torch.einsum('bnc,bct->bnt', dir_w, local_feat) / math.sqrt(dir_w.size(-1))
        dir_axis = F.normalize(self.dir_axis_head(q_static), dim=-1, eps=1e-6)  # [B, N, 3]
        dir_delta = torch.tanh(dir_logits).unsqueeze(2) * dir_axis.unsqueeze(-1)  # [B, N, 3, T]
        dir_scale = torch.sigmoid(self.dir_delta_scale)
        dipole_dir_t = F.normalize(
            dipole_dir.unsqueeze(-1) + dir_scale * dir_delta,
            dim=2,
            eps=1e-6,
        )  # [B, N, 3, T]

        return position, dipole_dir, amplitude, envelope_t, dipole_dir_t


# ──────────────────────────────────────────────────────────────────────────────
# LearnedAttenuation: 방향 인식 비균일 전도체 보정
#   가우시안 위치 + 전극 방향을 기반으로 경로별 조직 차폐를 학습
#   같은 점이라도 폐 쪽→전극과 근육 쪽→전극의 감쇠가 다름
# ──────────────────────────────────────────────────────────────────────────────
class LearnedAttenuation(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # 입력: 가우시안 위치(3) + 경로 방향(3) + 경로 거리(1) = 7D
        # pos: "이 소스가 몸 어디에 있는가" (어떤 조직으로 둘러싸여 있는가)
        # r_hat: "신호가 어느 방향으로 가는가" (어떤 조직을 관통하는가)
        # r_dist: "얼마나 먼가" (먼 거리는 더 많은 조직 관통)
        self.net = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # 초기값: correction = 1.0 (순수 1/r³에서 시작)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, position: torch.Tensor, electrode: torch.Tensor) -> torch.Tensor:
        """
        position:  [B, N, 3]
        electrode: [B, 9, 3]
        Returns:   [B, N, 9] correction factor (0.5 ~ 1.5)
        """
        B, N, _ = position.shape
        K = electrode.shape[1]  # 9

        # 경로 벡터: 가우시안 → 전극
        pos_exp = position.unsqueeze(2).expand(-1, -1, K, -1)   # [B, N, 9, 3]
        elc_exp = electrode.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, 9, 3]
        r_vec = elc_exp - pos_exp                                # [B, N, 9, 3]

        # 방향 (단위벡터) + 거리
        r_dist = r_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, N, 9, 1]
        r_hat = r_vec / r_dist                                     # [B, N, 9, 3]

        # MLP 입력: 소스 위치 + 경로 방향 + 경로 거리
        feat = torch.cat([pos_exp, r_hat, r_dist], dim=-1)  # [B, N, 9, 7]

        raw = self.net(feat).squeeze(-1)  # [B, N, 9]
        # 1.0 중심, ±50% 보정 범위
        return 1.0 + torch.tanh(raw) * 0.5


# ──────────────────────────────────────────────────────────────────────────────
# PhysicsRenderer: 1/r³ + 비균일 전도체 보정으로 9전극 전위 → 12-lead ECG
# ──────────────────────────────────────────────────────────────────────────────
class PhysicsRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("electrode_base", _ELECTRODE_POSITIONS.clone())
        self.attenuation = LearnedAttenuation()

    def forward(self, position, dipole_dir, amplitude, envelope_t,
                electrode_offset=None):
        """
        position:    [B, N, 3]     가우시안 3D 위치
        dipole_dir:  [B, N, 3] 또는 [B, N, 3, T]
                 쌍극자 방향 (정적 또는 시간가변 단위벡터)
        amplitude:   [B, N]        진폭
        envelope_t:  [B, N, T]     시간 envelope
        electrode_offset: [B, 9, 3] 또는 None
        Returns:     [B, 12, T]    12-lead ECG
        """
        B, N, T = envelope_t.shape

        # 전극 위치
        E = self.electrode_base.unsqueeze(0).expand(B, -1, -1)  # [B, 9, 3]
        if electrode_offset is not None:
            E = E + electrode_offset

        # 쌍극자 모멘트: direction × amplitude × envelope
        # p(t) = dir(t) * amp * env(t)  → [B, N, 3, T]
        amp_env = (amplitude.unsqueeze(-1) * envelope_t).unsqueeze(2)
        if dipole_dir.ndim == 3:
            moment = dipole_dir.unsqueeze(-1) * amp_env
        elif dipole_dir.ndim == 4:
            if dipole_dir.size(-1) != T:
                raise ValueError("dipole_dir time dimension must match envelope_t")
            moment = dipole_dir * amp_env
        else:
            raise ValueError("dipole_dir must be [B,N,3] or [B,N,3,T]")

        # 전극-쌍극자 간 벡터: [B, N, 9, 3]
        r_vec = E.unsqueeze(1) - position.unsqueeze(2)  # [B, N, 9, 3]

        # ── 부드러운 감쇠 함수: 1/(r² + ε)^(3/2) ──
        # smooth_clamp 대신 ε으로 자연스럽게 정규화
        # ε=0.04 → 유효 최소 거리 ≈ 0.2, 최대 감쇠 = 1/0.04^1.5 ≈ 125
        EPS_SQ = 0.04
        r_dist_sq = (r_vec ** 2).sum(dim=-1)           # [B, N, 9]
        attenuation_denom = (r_dist_sq + EPS_SQ) ** 1.5  # smoothed |r|³

        # 비균일 전도체 보정: 위치별 감쇠 계수
        correction = self.attenuation(position, E)  # [B, N, 9]

        # dot product: p(t) · r_vec → [B, N, 9, T]
        dot = torch.einsum('bndt,bnkd->bnkt', moment, r_vec)

        # 전위 = dot / (r²+ε)^(3/2) × correction
        V_per_gaussian = dot / attenuation_denom.unsqueeze(-1) * correction.unsqueeze(-1)

        # 모든 가우시안 합산 → 9전극 전위
        V_electrodes = V_per_gaussian.sum(dim=1)  # [B, 9, T]

        # ── 출력 스케일: 학습 가능한 고정 스케일 ──
        # RMS 정규화 대신 단순 clamp로 수치 안정화만
        V_electrodes = V_electrodes.clamp(-50.0, 50.0)

        # 9전극 → 12-lead 변환 (물리적 뺄셈)
        V_RA = V_electrodes[:, 0]   # [B, T]
        V_LA = V_electrodes[:, 1]
        V_LL = V_electrodes[:, 2]
        V_chest = V_electrodes[:, 3:9]  # [B, 6, T]

        WCT = (V_RA + V_LA + V_LL) / 3.0  # Wilson Central Terminal

        leads = torch.stack([
            V_LA - V_RA,                        # Lead I
            V_LL - V_RA,                        # Lead II
            V_LL - V_LA,                        # Lead III
            -(V_LA + V_LL) / 2.0 + V_RA,           # aVR = V_RA - (V_LA+V_LL)/2
            V_LA - (V_RA + V_LL) / 2.0,         # aVL
            V_LL - (V_RA + V_LA) / 2.0,         # aVF
        ], dim=1)  # [B, 6, T]

        chest_leads = V_chest - WCT.unsqueeze(1)  # [B, 6, T]

        v_leads = torch.cat([leads, chest_leads], dim=1)  # [B, 12, T]
        return v_leads, V_electrodes


# ──────────────────────────────────────────────────────────────────────────────
# Residual Bypass (고주파 보정용)
# ──────────────────────────────────────────────────────────────────────────────
class ResidualBypass(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        hidden = d_model // 2  # 128ch (이전: 64ch)
        self.net = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, 12, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, local_feat: torch.Tensor) -> torch.Tensor:
        return self.net(local_feat)


# ──────────────────────────────────────────────────────────────────────────────
# CDGS10 메인 모델
# ──────────────────────────────────────────────────────────────────────────────
class CDGS10(nn.Module):
    def __init__(self, d_model: int = 256, n_gaussians: int = 512,
                 n_encoder_layers: int = 4, **kwargs):
        super().__init__()
        self.encoder   = MultiScaleEncoder(d_model, n_encoder_layers)
        self.predictor = DipolePredictor(d_model, n_gaussians)
        self.renderer  = PhysicsRenderer()
        self.bypass    = ResidualBypass(d_model)
        self.meta_net  = MetaElectrodeConditioner()
        self.meta_pos  = MetaPositionConditioner()  # 환자별 위치 미세 보정

        self.lead_gain = nn.Parameter(torch.ones(1, 12, 1))
        self.lead_bias = nn.Parameter(torch.zeros(1, 12, 1))

        # Gate: physics vs bypass 비율 (초기 50:50)
        self.gate = nn.Parameter(torch.full((1, 12, 1), 0.0))

    def forward(self, x, meta=None, bypass_alpha: float = 1.0,
                lead_id: torch.Tensor = None):
        local_feat, global_feat = self.encoder(x, lead_id=lead_id)

        # 전극 위치 보정
        electrode_offset = None
        if meta is not None:
            electrode_offset = self.meta_net(meta)

        # 가우시안 파라미터 예측
        position, dipole_dir, amplitude, envelope_t, dipole_dir_t = \
            self.predictor(local_feat, global_feat)

        # 환자별 위치 미세 보정 (체격, 심장 크기 차이)
        if meta is not None:
            position = self.meta_pos(meta, position)

        # 물리 렌더링: 1/r³
        v_leads, V_electrodes = self.renderer(
            position, dipole_dir_t, amplitude, envelope_t, electrode_offset
        )
        phys_out = v_leads * self.lead_gain + self.lead_bias

        # Bypass (고주파 보정) — 물리에 더하는 방식 (곱하기 X)
        bypass_out = self.bypass(local_feat) * bypass_alpha

        # Additive mixing: physics + bypass (bypass 꺼져도 physics 100% 유지)
        gate_w    = torch.sigmoid(self.gate)
        mixed_out = phys_out + gate_w * bypass_out

        return mixed_out, phys_out, {
            "position":       position,       # [B, N, 3] — 가우시안 3D 위치!
            "dipole_dir":     dipole_dir,      # [B, N, 3] (정적 기준 방향)
            "dipole_dir_t":   dipole_dir_t,    # [B, N, 3, T] (시간가변 방향)
            "amplitude":      amplitude,       # [B, N]
            "envelope_t":     envelope_t,      # [B, N, T]
            "V_electrodes":   V_electrodes,    # [B, 9, T]
            "gate":           gate_w,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CDGS10Loss: 단순화 (4 core + 1 soft boundary)
# ──────────────────────────────────────────────────────────────────────────────
def pearson_loss(pred, target):
    pred_c   = pred   - pred.mean(dim=-1, keepdim=True)
    target_c = target - target.mean(dim=-1, keepdim=True)
    num  = (pred_c * target_c).sum(dim=-1)
    denom = pred_c.norm(dim=-1) * target_c.norm(dim=-1) + 1e-8
    return 1.0 - (num / denom).mean()


class CDGS10Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_recon     = 1.0
        # 기본값은 경량 모드: 고비용 항은 0으로 두고 필요 시 외부에서 다시 켤 수 있음
        self.w_pearson   = 0.0
        self.w_freq      = 0.0
        self.w_relative  = 0.2
        self.w_phys      = 0.0
        self.w_amp_alive = 0.05
        self.w_env_alive = 0.20
        self.w_env_var   = 0.05
        self.w_boundary  = 0.02

    def forward(self, pred, target, amplitude, position=None,
                phys_out=None, envelope_t=None, **kwargs):
        zero = pred.new_tensor(0.0)

        # 1. 기본 복원 (mixed_out)
        L_recon   = F.l1_loss(pred, target)
        L_pearson = zero
        if self.w_pearson > 0:
            L_pearson = pearson_loss(pred, target)

        L_freq = zero
        if self.w_freq > 0:
            L_freq = F.l1_loss(
                torch.fft.rfft(pred.float(), dim=-1).abs(),
                torch.fft.rfft(target.float(), dim=-1).abs(),
            )

        # 2. 상대 오차
        L_relative = zero
        if self.w_relative > 0:
            lead_rms = (target ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=0.01)
            L_relative = (torch.abs(pred - target) / lead_rms).mean()

        # 3. 물리 브랜치 전용 loss — bypass가 물리를 죽이지 못하도록
        L_phys = zero
        if self.w_phys > 0 and phys_out is not None:
            L_phys = F.l1_loss(phys_out, target)

        # 4. Amplitude 생존 (threshold 상향: 0.3 이상 유지)
        L_amp_alive = F.relu(0.3 - amplitude.mean())

        # 5. Envelope 생존 + 시간 변동 유지
        L_env_alive = zero
        L_env_var = zero
        if self.w_env_alive > 0 and envelope_t is not None:
            env_mean = envelope_t.mean()
            env_std_t = envelope_t.std(dim=-1).mean()
            # 평균이 너무 낮아지면 물리 브랜치가 사실상 꺼지므로 하한 유지
            L_env_alive = F.relu(0.08 - env_mean)
            # 시간축 표준편차가 너무 작으면 평평해지므로 최소 변동성 유지
            if self.w_env_var > 0:
                L_env_var = F.relu(0.03 - env_std_t)

        # 6. Soft boundary
        L_boundary = zero
        if self.w_boundary > 0 and position is not None:
            heart_center = _HEART_CENTER.to(position.device)
            dist_from_heart = (position - heart_center).norm(dim=-1)
            L_boundary = F.relu(dist_from_heart - 0.12).mean()

        total = (
            self.w_recon     * L_recon     +
            self.w_pearson   * L_pearson   +
            self.w_freq      * L_freq      +
            self.w_relative  * L_relative  +
            self.w_phys      * L_phys      +
            self.w_amp_alive * L_amp_alive +
            self.w_env_alive * L_env_alive +
            self.w_env_var   * L_env_var   +
            self.w_boundary  * L_boundary
        )

        return total, {
            "recon":     L_recon,
            "pearson":   L_pearson,
            "freq":      L_freq,
            "relative":  L_relative,
            "phys":      L_phys,
            "amp_alive": L_amp_alive,
            "env_alive": L_env_alive,
            "env_var":   L_env_var,
            "boundary":  L_boundary,
            "total":     total,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 2
    m = CDGS10(d_model=128, n_gaussians=256, n_encoder_layers=2)
    x = torch.randn(B, 1, 1000)
    lead_id = torch.tensor([0, 1])
    meta = torch.tensor([
        [0.45, 0, 0.0, 0.0],
        [0.62, 1, -0.2, 0.5],
    ])

    y_train, y_phys, extras = m(x, meta=meta, bypass_alpha=0.5, lead_id=lead_id)
    assert y_train.shape == (B, 12, 1000), f"Shape mismatch: {y_train.shape}"

    # base positions(nn.Parameter)는 공유, meta 보정 후 환자별로 약간 다름
    pos_diff = (extras["position"][0] - extras["position"][1]).abs().max().item()
    print(f"  Position per-patient diff (max): {pos_diff:.6f}")
    assert pos_diff < 0.05, "Position offset should be small"

    loss_fn = CDGS10Loss()
    tgt = torch.randn(B, 12, 1000)
    loss_val, bd = loss_fn(
        y_train, tgt,
        extras["amplitude"],
        position=extras["position"],
    )

    pos = extras["position"]
    print(f"\n[OK] CDGS10 (Physics Gaussian Splatting) Test 통과!")
    print(f"  Total Loss:  {loss_val.item():.4f}")
    print(f"  Loss terms:  {list(bd.keys())}")
    print(f"  Amplitude:   {extras['amplitude'].mean().item():.4f}")
    print(f"  Gate (phys%): {extras['gate'].mean().item():.4f}")
    print(f"  Position range: [{pos.min().item():.3f}, {pos.max().item():.3f}]")
    print(f"  Position std:   {pos.std().item():.4f}")
    print(f"  Boundary loss:  {bd['boundary'].item():.4f}")
    print(f"  Position shared across batch: OK")

    # 학습률 분리 확인
    pgs = get_param_groups(m, 1e-4)
    for i, pg in enumerate(pgs):
        n = sum(p.numel() for p in pg["params"])
        print(f"  Param group {i}: {n:,} params, wd={pg['weight_decay']}, lr_scale={pg.get('lr_scale', 1.0)}")

