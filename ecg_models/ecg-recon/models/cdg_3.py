# -*- coding: utf-8 -*-
"""
CDGS v3 (`cdg_3.py`): **`CDGS3`** 클래스.
`cdg_2.py`와 동일 시점 스냅샷에서 분리한 포크 — `--model cdg_3` 로 학습.
v3 실험은 이 파일만 수정하면 됩니다.

──────────────────────────────────────────────────────────────────────────────
[각주] N개 공통 4D 가우시안 → 리드 k 축 L_k(t) 평면 투영·가중 스플랫
──────────────────────────────────────────────────────────────────────────────
예측기 출력은 리드와 무관한 N슬롯 공통이다. 슬롯 n마다 시간 봉우리 g_n(t),
구 내부 3D 점 p_pos_n(t), 구면조화 계수의 시간 변화 sh_n(t)가 있다.

리드 k에 대해서만 다음이 달라진다. 단위 벡터 L_k(t)를 카메라 광축으로 두고,
L_k에 수직인 평면에 정규직교 기저 (e1, e2)를 잡는다 (`camera_plane_basis`).
직교 투영 좌표는 u = (p_pos·e1), v = (p_pos·e2) 이고, 격자 스케일을 위해 tanh로
대략 [-1, 1] 근처로 보낸다.

고정 격자 (H×W)의 각 픽셀은 평면상 (u_grid, v_grid)에 대응한다. 슬롯 n이
(u_n, v_n)에 가우시안 blob을 남기듯, 픽셀까지의 (u,v) 거리로 2D 커널 가중을 준다.
채널 0에는 g_n(t)·커널, 채널 1에는 (g_n·sh_proj)·커널을 누적하는데, 이때
앞쪽 반구 게이트와 (dipole 방향 vs L_k) 각도 기반 dist_atten을 곱한 weight로
진폭을 줄인다 → 이게 “가중 스플랫”. 리드 k·시각 t마다 (2, H, W) 맵 하나가 되고,
공유 `SharedSplatReadout` CNN이 스칼라 V_k(t)로 눌러 12리드를 만든다.

──────────────────────────────────────────────────────────────────────────────
[각주] Docker — **아래 각 줄을 통째로 복사** (백슬래시 줄 이어쓰기 없음)
레포 루트가 컨테이너 `/workspace`에 붙었다고 가정. GPU 번호·데이터 호스트 경로·
epochs·batch_size 만 본인 환경에 맞게 고친다. OOM이면 --batch_size 4 등.
direct 학습이면 train 명령 끝에 `--cdgs_direct_lead_sum` 등 기존과 동일하게 추가.

① 학습 — **백그라운드**. `-v` **왼쪽** 경로만 본인 서버의 PTB-XL 디렉터리로 바꿈
   (PhysioNet 받은 `ptb-xl/1.0.3` 폴더 — 안에 `records100` 등 있어야 함). 오른쪽 `:/workspace/data/ptb-xl` 은 고정.
   예: `-v ~/project_jyu/physionet.org/files/ptb-xl/1.0.3:/workspace/data/ptb-xl`
   GPU는 `--gpus '"device=6"'` 처럼 번호만 수정. `</dev/null` 은 nohup 안내 줄을 로그에 안 남기려는 용도.
nohup docker run --rm --gpus '"device=6"' -v ~/project_jyu/physionet.org/files/ptb-xl/1.0.3:/workspace/data/ptb-xl -v "$(pwd)":/workspace -w /workspace pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime bash -c "pip install -q wfdb pandas scipy scikit-learn tqdm 'numpy<2' && python ecg_1to12/train_all_models.py --model cdg_3 --data_dir ./data/ptb-xl --epochs 30 --batch_size 4 --skip_download" > cdg_3_train.log 2>&1 </dev/null &
로그 확인: tail -f cdg_3_train.log

② 이 파일만 스모크 (forward 한 번, 이미지에 torch 포함):
docker run --rm --gpus '"device=6"' -v "$(pwd)":/workspace -w /workspace pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime bash -c "python ecg_models/models/cdg_3.py"

③ Windows PowerShell 예: `$(pwd)` 대신 `-v C:/Users/이름/프로젝트:/workspace` (백그라운드는 Git Bash/WSL에서 ① 또는 ④ 권장)
④ 같은 백그라운드를 스크립트로: `MODEL=cdg_3 BATCH_SIZE=4 GPU_DEVICE=0 ./ecg_1to12/run_docker_train_nohup.sh` (데이터 경로는 `PTBXL_HOST` 로 지정)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# 해부학적 리드 벡터 세팅
# ═══════════════════════════════════════════════════════════════════════════
# (참고) 고정 6축 초기화에 쓰이던 값 — 사지 축은 이제 아래 삼각형 파생만 사용.
_LIMB_VECTORS_RAW = [
    [ 1.000,  0.000,  0.000],  # I
    [ 0.500,  0.866,  0.000],  # II
    [-0.500,  0.866,  0.000],  # III
    [-0.866, -0.500,  0.000],  # aVR
    [ 0.866, -0.500,  0.000],  # aVL
    [ 0.000,  1.000,  0.000],  # aVF
]
_CHEST_VECTORS_RAW = [
    [-0.707,  0.000,  0.707],  # V1
    [ 0.000,  0.000,  1.000],  # V2
    [ 0.438,  0.000,  0.899],  # V3
    [ 0.707,  0.000,  0.707],  # V4
    [ 0.899,  0.000,  0.438],  # V5
    [ 0.966,  0.000,  0.259],  # V6
]
LIMB_VECTORS = F.normalize(torch.tensor(_LIMB_VECTORS_RAW, dtype=torch.float32), dim=-1)
CHEST_VECTORS_INIT = F.normalize(torch.tensor(_CHEST_VECTORS_RAW, dtype=torch.float32), dim=-1)
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def fibonacci_unit_sphere(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """단위구 위에 n 점을 황금나선으로 균등 배치 → (n, 3)."""
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
    """
    p0, p_vel: (B, N, 3)  /  dt: (B, N, T)
    p_raw = p0 + p_vel * dt → 단위구 *내부*: 방향(p_raw/||p_raw||) * tanh(||p_raw||) ∈ open unit ball.
    렌더러·반구 게이트·거리 감쇠는 이 3D 점을 카메라 평면에 투영해 사용.
    """
    p_raw = p0.unsqueeze(-1) + p_vel.unsqueeze(-1) * dt.unsqueeze(2)
    nrm = p_raw.norm(dim=2, keepdim=True).clamp_min(1e-8)
    return (p_raw / nrm) * torch.tanh(nrm)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Encoder Blocks
# ═══════════════════════════════════════════════════════════════════════════
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        ch_per = out_ch // len(dilations)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, ch_per, kernel_size=5, padding=d * 2, dilation=d) for d in dilations
        ])
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
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model*4, batch_first=True) 
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

# ═══════════════════════════════════════════════════════════════════════════
# 2. Gaussian Predictor (4D: time-varying σ(t), moving 3D point p(t) + SH)
# ═══════════════════════════════════════════════════════════════════════════
class GaussianPredictor(nn.Module):
    """
    Per Gaussian:
      - mu: time center (normalized [0,1])
      - sigma0, sigma_vel: σ(t) = softplus(sigma0 + sigma_vel * dt) + eps, dt = t - mu
      - amplitude
      - sh_base, sh_vel: SH coeffs drift in time (angular pattern)
      - p0_delta, p_vel: p0 = p0_anchor + p0_delta (단위구 밖일 수 있음) → 렌더에서
        tanh(||p_raw||)·방향 으로 **단위구 내부** 3D 점으로 맵.
    """

    def __init__(self, d_model=256, n_gaussians=64, n_heads=8):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.sh_dim = 9  # Degree 2
        out_dim = 4 + self.sh_dim * 2 + 3 + 3  # mu, sig0, sig_vel, amp, sh×2, p0_delta, p_vel
        # 쿼리: 단위 하이퍼구면에 투영해 슬롯마다 초기 방향을 고르게
        _gq = torch.randn(1, n_gaussians, d_model)
        self.gaussian_queries = nn.Parameter(F.normalize(_gq, dim=-1, eps=1e-6))
        # p0 앵커: 방향은 황금나선, 반지름 ~0.35 로 **구 내부**에서 시작
        _p0 = fibonacci_unit_sphere(n_gaussians, device=_gq.device, dtype=_gq.dtype).unsqueeze(0)
        self.p0_anchor = nn.Parameter((_p0 * 0.35).clone())
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.param_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_dim),
        )
        nn.init.zeros_(self.param_head[-1].bias)
        nn.init.normal_(self.param_head[-1].weight, std=0.02)

    def forward(self, global_feat):
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
        p_vel = params[..., 25:28]

        p0 = self.p0_anchor.expand(B, -1, -1) + p0_delta
        return mu, sigma0, sigma_vel, amplitude, sh_base, sh_vel, p0, p_vel

# ═══════════════════════════════════════════════════════════════════════════
# 3. Metadata Conditioner (Dynamic)
# ═══════════════════════════════════════════════════════════════════════════
class MetadataConditioner(nn.Module):
    def __init__(self, d_meta=32):
        super().__init__()
        self.age_proj = nn.Linear(1, d_meta // 2)
        self.sex_embed = nn.Embedding(2, d_meta // 2)
        self.chest_mlp = nn.Sequential(nn.Linear(d_meta, d_meta), nn.GELU(), nn.Linear(d_meta, 18))
        self.resp_mlp = nn.Sequential(nn.Linear(d_meta, d_meta), nn.GELU(), nn.Linear(d_meta, 4))

    def forward(self, age, sex):
        meta = torch.cat([self.age_proj(age.unsqueeze(-1).float()), self.sex_embed(sex.long())], dim=-1)
        chest_offset = self.chest_mlp(meta).view(-1, 6, 3) * 0.1
        resp_params = self.resp_mlp(meta)
        return chest_offset, torch.sigmoid(resp_params[..., 0]) * 5.0, resp_params[..., 1:4] * 0.05

# ═══════════════════════════════════════════════════════════════════════════
# 4. 카메라 평면 (u,v) + 2D 스플랫 → 리드별 학습 CNN (4DGS 스타일)
# ═══════════════════════════════════════════════════════════════════════════
def camera_plane_basis(L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """L: (B,K,3,T) 단위벡터 → 카메라 이미지 평면 정규직교기저 e1, e2 (각도, B,K,3,T)."""
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
    """12 리드 공통: 2채널 스플랫 맵 (B*T,2,H,W) → 스칼라 (국소 패치 → AdaptiveAvgPool → fc)."""

    def __init__(self, in_ch: int = 2, hidden: int = 32, grid: int = 8):
        super().__init__()
        self.grid = grid
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (M, in_ch, H, W)
        h = self.net(x).flatten(1)
        return self.fc(h)


class GaussianRenderer(nn.Module):
    """
    4DGS에 가깝게:
      3) 채널0: 위치 스플랫, 채널1: 같은 (u,v)에 SH 기여를 가중한 스플랫.
      4) **공통 SharedSplatReadout** 한 개로 모든 리드에 동일 가중치 적용:
      (B*T, 2, H, W) → 스칼라 → V_k(t) (direct / full grid 동일).

    **3D dipole 위치**: p_raw = p0 + p_vel*Δt 후 **normalize 없이**
      p = (p_raw/||p_raw||) * tanh(||p_raw||) 로 **단위구 내부**에 두고,
      각 리드 평면에 **직교 투영 (u,v)** → 2D 스플랫 합산.

    direct_lead_sum=True 이면 (레이저 + 국소 패치 모드):
      원점 근처 작은 (u,v) 패치만 사용; 읽기는 **합산이 아니라** SharedSplatReadout.

    **거리 감쇠**: 리드 축 L과 dipole 방향 p̂ 사이 구면 각 θ = arccos(p·L)에 대해
      exp(-γ θ²) (γ=softplus(학습 파라미터))로 기여 감소 — “전극–원 거리/방향” 둔감 모델링.

    lead_halfspace_gate=True 이면, 리드 축 L을 “보는 방향”으로 두고 p·L ≤ 0 인
    (축 뒤쪽 반구) 가우시안은 해당 리드 스플랫에 **기여하지 않음** (relu(p·L) 가중).

    사지 6유도: **RA·LA·LL** 세 점을 nn.Parameter 로 두고,
    I/II/III 및 Goldberger 증폭 aVR/aVL/aVF 축을 **삼각형 기하로 매 forward 파생** (교육용
      1) 각 리드 k = 카메라 축 L_k(t). 점 p_hat(t)를 L에 수직인 평면에 (u,v)로 투영.
      2) N개 가우시안을 그 평면의 H×W 격자에 스플랫 (진폭×가우시안 커널). 그림과 동일 논리).
    RA 초깃값 0에서 시작해 살짝 움직일 수 있음(전체 평행이동은 사지 6축에 상쇄되지만, 삼각형 형태는 세 점 모두에 의존).
    """

    def __init__(
        self,
        splat_grid: int = 8,
        cnn_hidden: int = 32,
        lead_halfspace_gate: bool = True,
        direct_lead_sum: bool = False,
        direct_lead_patch: int = 5,
        direct_lead_span: float = 0.55,
    ):
        super().__init__()
        self.splat_grid = splat_grid
        self.lead_halfspace_gate = lead_halfspace_gate
        self.direct_lead_sum = direct_lead_sum
        self.direct_lead_patch = int(direct_lead_patch)
        self.direct_lead_span = float(direct_lead_span)
        H = W = splat_grid
        self.register_buffer("grid_u", torch.linspace(-1.0, 1.0, W).view(1, 1, 1, 1, W))
        self.register_buffer("grid_v", torch.linspace(-1.0, 1.0, H).view(1, 1, 1, H, 1))
        self.splat_tau_raw = nn.Parameter(torch.tensor(0.0))
        # 구면 각 기반 감쇠 강도 (양수); 작게 시작
        self.dist_atten_raw = nn.Parameter(torch.tensor(-0.5))
        # 아인토벤 삼각형: RA≈0, LA=(1,0), LL=(1/2,√3/2) 에서 시작 → RA도 학습
        self.limb_ra = nn.Parameter(torch.zeros(3))
        self.limb_la = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
        self.limb_ll = nn.Parameter(torch.tensor([0.5, math.sqrt(3) / 2, 0.0]))
        self.chest_vectors = nn.Parameter(CHEST_VECTORS_INIT.clone())
        self.shared_splat_readout = SharedSplatReadout(in_ch=2, hidden=cnn_hidden, grid=splat_grid)

    def limb_vectors(self) -> torch.Tensor:
        """(6,3) 단위벡터: I, II, III, aVR, aVL, aVF. RA·LA·LL 학습, 축은 항상 공식 파생."""
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

    def forward(
        self,
        mu,
        sigma0,
        sigma_vel,
        amplitude,
        sh_base,
        sh_vel,
        p0,
        p_vel,
        T,
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
        t = torch.linspace(0, 1, T, device=mu.device).view(1, 1, T)
        mu_bc = mu.unsqueeze(-1)
        dt = t - mu_bc
        sigma_t = F.softplus(sigma0.unsqueeze(-1) + sigma_vel.unsqueeze(-1) * dt) + 1e-3
        gauss = amplitude.unsqueeze(-1) * torch.exp(-0.5 * (dt / sigma_t) ** 2)

        sh_dynamic = sh_base.unsqueeze(-1) + sh_vel.unsqueeze(-1) * dt.unsqueeze(2)

        p_pos = dipole_position_in_unit_ball(p0, p_vel, dt)

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

        # p_pos(구 내부)와 단위 L → 코사인 유사도 = (p·L)/||p||
        p_nrm = p_pos.norm(dim=2, keepdim=True).clamp_min(1e-8)
        cos_p_L = torch.einsum("bnct,bkct->bnkt", p_pos, lead_vecs_t) / p_nrm

        splat_pos = torch.zeros(B, 12, T, H, W, device=mu.device, dtype=mu.dtype)
        splat_sh = torch.zeros(B, 12, T, H, W, device=mu.device, dtype=mu.dtype)

        gamma = F.softplus(self.dist_atten_raw) + 1e-6

        for n in range(N):
            u_n = u[:, n]
            v_n = v[:, n]
            g_n = gauss[:, n]
            sh_n = sh_proj[:, n]
            du = u_n.unsqueeze(-1).unsqueeze(-1) - gx
            dv = v_n.unsqueeze(-1).unsqueeze(-1) - gy
            kernel = torch.exp(-0.5 * (du * du + dv * dv) / (tau * tau))
            g_expand = g_n.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
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

        outs = []
        for k in range(12):
            inp = torch.stack([splat_pos[:, k], splat_sh[:, k]], dim=2).reshape(B * T, 2, H, W)
            outs.append(self.shared_splat_readout(inp))
        V = torch.cat(outs, dim=1).view(B, 12, T)
        return V

# ═══════════════════════════════════════════════════════════════════════════
# 5. Residual Refiner (간단한 CNN)
# ═══════════════════════════════════════════════════════════════════════════
class ResidualRefiner(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(13, base_ch, 3, padding=1), nn.GELU(),
            nn.Conv1d(base_ch, 12, 1)
        )
    def forward(self, lead_I, coarse):
        return coarse + self.net(torch.cat([lead_I, coarse], dim=1))

# ═══════════════════════════════════════════════════════════════════════════
# 6. Main CDGS3 Class (cdgs v3)
# ═══════════════════════════════════════════════════════════════════════════
class CDGS3(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_gaussians=128,
        n_encoder_layers=4,
        use_metadata=True,
        lead_halfspace_gate: bool = True,
        direct_lead_sum: bool = False,
        direct_lead_patch: int = 5,
        direct_lead_span: float = 0.55,
    ):
        super().__init__()
        self.use_metadata = use_metadata
        self.encoder = MultiScaleEncoder(d_model, n_encoder_layers)
        self.gaussian_predictor = GaussianPredictor(d_model, n_gaussians)
        self.meta_conditioner = MetadataConditioner() if use_metadata else None
        self.renderer = GaussianRenderer(
            lead_halfspace_gate=lead_halfspace_gate,
            direct_lead_sum=direct_lead_sum,
            direct_lead_patch=direct_lead_patch,
            direct_lead_span=direct_lead_span,
        )
        self.refiner = ResidualRefiner()

    def forward(self, x, age=None, sex=None):
        B, _, T = x.shape
        local_feat, global_feat = self.encoder(x)
        mu, sigma0, sigma_vel, amplitude, sh_base, sh_vel, p0, p_vel = self.gaussian_predictor(
            global_feat
        )

        chest_offset, resp_freq, resp_vec = None, None, None
        if self.use_metadata and age is not None and sex is not None:
            chest_offset, resp_freq, resp_vec = self.meta_conditioner(age, sex)

        pred_coarse = self.renderer(
            mu,
            sigma0,
            sigma_vel,
            amplitude,
            sh_base,
            sh_vel,
            p0,
            p_vel,
            T=T,
            chest_offset=chest_offset,
            resp_freq=resp_freq,
            resp_vec=resp_vec,
        )
        pred_fine = self.refiner(x, pred_coarse)

        return pred_fine, pred_coarse, {
            "mu": mu,
            "sigma0": sigma0,
            "sigma_vel": sigma_vel,
            "amplitude": amplitude,
            "p0": p0,
            "p_vel": p_vel,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 7. CDGS3Loss + describe_gaussians
# ═══════════════════════════════════════════════════════════════════════════
class CDGS3Loss(nn.Module):
    """
    CDGS3 전용 손실. 순수 L1만 쓰면 QRS 같은 날카로운 피크·깊은 음의 편향이
    상대적으로 약해져 평활하게 나오기 쉬움 → CDGSLoss와 동일한
    morphology(진폭 큰 시점 가중)·freq 항을 포함.
    train_all_models.py: cdg_3 일 때만 CDGS3Loss() 사용.
    네 번째 인자(amplitude)는 호환용(미사용).
    """

    def __init__(
        self,
        w_recon: float = 1.0,
        w_coarse: float = 0.3,
        w_morph: float = 0.25,
        w_freq: float = 0.1,
    ):
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
        pred_fft = torch.fft.rfft(p, dim=-1)
        target_fft = torch.fft.rfft(t, dim=-1)
        return F.l1_loss(pred_fft.abs(), target_fft.abs())

    def morphology_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # |target|이 큰 곳(양봉·음봉 모두)에 더 큰 패널티 → 피크/깊은 음수 구간 보강
        weights = 1.0 + target.abs() / (target.abs().mean(dim=-1, keepdim=True) + 1e-6)
        return (weights * (pred - target).abs()).mean()

    def forward(
        self,
        pred_fine: torch.Tensor,
        pred_coarse: torch.Tensor,
        target: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        del amplitude  # 호환용
        losses = {}
        losses["recon"] = self.reconstruction_loss(pred_fine, target)
        losses["coarse"] = self.reconstruction_loss(pred_coarse, target)
        losses["morph"] = self.morphology_loss(pred_fine, target)
        losses["freq"] = self.frequency_loss(pred_fine, target)
        total = (
            self.w_recon * losses["recon"]
            + self.w_coarse * losses["coarse"]
            + self.w_morph * losses["morph"]
            + self.w_freq * losses["freq"]
        )
        losses["total"] = total
        return total, losses


def describe_gaussians(extras: dict, T: int = 1000) -> dict:
    """
    extras: mu, sigma0, sigma_vel, amplitude, p0, p_vel (sigma는 시간함수 σ(t)).
    T 샘플이 10초 구간이면 mu*10000 ≈ ms 위치로 해석.
    """
    mu = extras["mu"][0].detach().cpu().numpy()
    amplitude = extras["amplitude"][0].detach().cpu().numpy()
    out = {
        "mu_ms": mu * 10000.0,
        "amplitude": amplitude,
    }
    if "sigma0" in extras:
        s0 = extras["sigma0"][0].detach().cpu().numpy()
        out["sigma0_raw"] = s0
        out["sigma_at_center_ms"] = (F.softplus(torch.tensor(s0)) + 1e-3).numpy() * 10000.0
    if "sigma_vel" in extras:
        out["sigma_vel"] = extras["sigma_vel"][0].detach().cpu().numpy()
    if "p0" in extras:
        out["p0"] = extras["p0"][0].detach().cpu().numpy()
    if "p_vel" in extras:
        out["p_vel"] = extras["p_vel"][0].detach().cpu().numpy()
    if "direction" in extras and "lead_vectors" in extras:
        direction = extras["direction"][0].detach().cpu().numpy()
        lead_vecs = extras["lead_vectors"][0].detach().cpu().numpy()
        projections = direction @ lead_vecs.T
        out["dominant_lead"] = [LEAD_NAMES[i] for i in projections.argmax(axis=1)]
        out["projections"] = projections
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 빠른 테스트
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    B, T = 2, 1000
    x = torch.randn(B, 1, T)
    age, sex = torch.tensor([0.55, 0.32]), torch.tensor([1, 0])
    
    model = CDGS3(d_model=256, n_gaussians=128, use_metadata=True)
    
    print(f"CDGS3 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    pred_fine, pred_coarse, extras = model(x, age, sex)
    print(f"pred_fine:   {pred_fine.shape}")
    print(f"pred_coarse: {pred_coarse.shape}")
    print("에러 없이 정상적으로 4D SH 렌더링 완료!")