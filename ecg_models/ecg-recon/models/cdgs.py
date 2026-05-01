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
      - p
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
        sh_base = params[..., 4:13]0_delta, p_vel: p(t) = normalize(p0_anchor + p0_delta + p_vel * dt)  (learnable anchor)
    """

    def __init__(self, d_model=256, n_gaussians=64, n_heads=8):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.sh_dim = 9  # Degree 2
        out_dim = 4 + self.sh_dim * 2 + 3 + 3  # mu, sig0, sig_vel, amp, sh×2, p0_delta, p_vel
        self.gaussian_queries = nn.Parameter(torch.randn(1, n_gaussians, d_model))
        # 학습 가능한 3D 점 초기 앵커 (슬롯마다) — 학습되며 움직임의 기준
        self.p0_anchor = nn.Parameter(torch.randn(1, n_gaussians, 3) * 0.05)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        sh_vel = params[..., 13:22]
        p0_delta = params[..., 22:25]
        p_vel = params[..., 25:28]

        p0 = F.normalize(self.p0_anchor.expand(B, -1, -1) + p0_delta, dim=-1, eps=1e-6)
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


class PerLeadSplatCNN(nn.Module):
    """한 리드(카메라) 전용: 2채널 스플랫 맵 → 스칼라 (해당 시간 샘플 1개)."""

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
      1) 각 리드 k = 카메라 축 L_k(t). 점 p_hat(t)를 L에 수직인 평면에 (u,v)로 투영.
      2) N개 가우시안을 그 평면의 H×W 격자에 스플랫 (진폭×가우시안 커널).
      3) 채널0: 위치 스플랫, 채널1: 같은 (u,v)에 SH 기여를 가중한 스플랫.
      4) 리드마다 **별도 학습 CNN**으로 (B*T, 2, H, W) → 스칼라 → V_k(t).

    direct_lead_sum=True 이면 (레이저 + 국소 패치 모드):
      전 평면 [-1,1]² 격자 대신 **원점 근처 작은 (u,v) 패치**만 사용
      (`direct_lead_patch`×`direct_lead_patch`, 좌표는 ±`direct_lead_span`).
      각 가우시안은 그 패치 셀에 2D 커널로 기여를 쌓고, **패치 전체를 합산**해
      **splat_pos + splat_sh → V_k(t)** (PerLeadSplatCNN 없음).

    lead_halfspace_gate=True 이면, 리드 축 L을 “보는 방향”으로 두고 p·L ≤ 0 인
    (축 뒤쪽 반구) 가우시안은 해당 리드 스플랫에 **기여하지 않음** (relu(p·L) 가중).

    사지 6유도: **RA·LA·LL** 세 점을 nn.Parameter 로 두고,
    I/II/III 및 Goldberger 증폭 aVR/aVL/aVF 축을 **삼각형 기하로 매 forward 파생** (교육용 그림과 동일 논리).
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
        # 아인토벤 삼각형: RA≈0, LA=(1,0), LL=(1/2,√3/2) 에서 시작 → RA도 학습
        self.limb_ra = nn.Parameter(torch.zeros(3))
        self.limb_la = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
        self.limb_ll = nn.Parameter(torch.tensor([0.5, math.sqrt(3) / 2, 0.0]))
        self.chest_vectors = nn.Parameter(CHEST_VECTORS_INIT.clone())
        self.lead_cnns = nn.ModuleList(
            [PerLeadSplatCNN(in_ch=2, hidden=cnn_hidden, grid=splat_grid) for _ in range(12)]
        )

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

        p_t = p0.unsqueeze(-1) + p_vel.unsqueeze(-1) * dt.unsqueeze(2)
        p_hat = F.normalize(p_t, dim=2, eps=1e-6)

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
        u = torch.einsum("bnct,bkct->bnkt", p_hat, e1)
        v = torch.einsum("bnct,bkct->bnkt", p_hat, e2)
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

        # p_hat, L 단위벡터 → cos = p·L. L과 같은 반구(“축 앞쪽”)에서만 스플랫 가중.
        cos_p_L = torch.einsum("bnct,bkct->bnkt", p_hat, lead_vecs_t)

        splat_pos = torch.zeros(B, 12, T, H, W, device=mu.device, dtype=mu.dtype)
        splat_sh = torch.zeros(B, 12, T, H, W, device=mu.device, dtype=mu.dtype)

        for n in range(N):
            u_n = u[:, n]
            v_n = v[:, n]
            g_n = gauss[:, n]
            sh_n = sh_proj[:, n]
            du = u_n.unsqueeze(-1).unsqueeze(-1) - gx
            dv = v_n.unsqueeze(-1).unsqueeze(-1) - gy
            kernel = torch.exp(-0.5 * (du * du + dv * dv) / (tau * tau))
            g_expand = g_n.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            if self.lead_halfspace_gate:
                hem = torch.clamp_min(cos_p_L[:, n], 0.0).unsqueeze(-1).unsqueeze(-1)
            else:
                hem = 1.0
            splat_pos = splat_pos + g_expand * kernel * hem
            w_sh = (g_n.unsqueeze(1) * sh_n).unsqueeze(-1).unsqueeze(-1)
            splat_sh = splat_sh + w_sh * kernel * hem

        if self.direct_lead_sum:
            # (B, 12, T) — 원점 주변 패치 셀 전부 합산 (레이저 축 근처 국소 적분)
            V = (splat_pos + splat_sh).sum(dim=-1).sum(dim=-1)
        else:
            outs = []
            for k in range(12):
                inp = torch.cat([splat_pos[:, k], splat_sh[:, k]], dim=1)
                inp = inp.reshape(B * T, 2, H, W)
                outs.append(self.lead_cnns[k](inp))
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
# 6. Main CDGS Class (수정된 몸통)
# ═══════════════════════════════════════════════════════════════════════════
class CDGS(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_gaussians=64,
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
# 7. CDGSLoss + describe_gaussians (학습 스크립트 / __init__ 호환)
# ═══════════════════════════════════════════════════════════════════════════
class CDGSLoss(nn.Module):
    """
    train_all_models.py: loss, _ = CDGSLoss()(pred_fine, pred_coarse, y, extras['amplitude'])
    """

    def __init__(
        self,
        w_recon: float = 1.0,
        w_coarse: float = 0.3,
        w_freq: float = 0.1,
        w_morph: float = 0.2,
        w_einthoven: float = 0.5,
        w_sparse: float = 0.001,
        w_temporal: float = 0.1,
    ):
        super().__init__()
        self.w_recon = w_recon
        self.w_coarse = w_coarse
        self.w_freq = w_freq
        self.w_morph = w_morph
        self.w_einthoven = w_einthoven
        self.w_sparse = w_sparse
        self.w_temporal = w_temporal

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)

    def frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # AMP(FP16) + cuFFT: non-power-of-two length (e.g. T=1000) is not supported in half.
        p = pred.float()
        t = target.float()
        pred_fft = torch.fft.rfft(p, dim=-1)
        target_fft = torch.fft.rfft(t, dim=-1)
        return F.l1_loss(pred_fft.abs(), target_fft.abs())

    def morphology_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + target.abs() / (target.abs().mean(dim=-1, keepdim=True) + 1e-6)
        return (weights * (pred - target).abs()).mean()

    def einthoven_loss(self, pred: torch.Tensor) -> torch.Tensor:
        I, II, III = pred[:, 0, :], pred[:, 1, :], pred[:, 2, :]
        aVR, aVL, aVF = pred[:, 3, :], pred[:, 4, :], pred[:, 5, :]
        l1 = F.l1_loss(III, II - I)
        l2 = F.l1_loss(aVR, -(I + II) / 2.0)
        l3 = F.l1_loss(aVL, (I - III) / 2.0)
        l4 = F.l1_loss(aVF, (II + III) / 2.0)
        return (l1 + l2 + l3 + l4) / 4.0

    def sparse_loss(self, amplitude: torch.Tensor) -> torch.Tensor:
        return amplitude.mean()

    def temporal_consistency_loss(self, pred: torch.Tensor) -> torch.Tensor:
        diff = pred[:, :, 1:] - pred[:, :, :-1]
        return diff.abs().mean()

    def forward(
        self,
        pred_fine: torch.Tensor,
        pred_coarse: torch.Tensor,
        target: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        losses = {}
        losses["recon"] = self.reconstruction_loss(pred_fine, target)
        losses["coarse"] = self.reconstruction_loss(pred_coarse, target)
        losses["freq"] = self.frequency_loss(pred_fine, target)
        losses["morph"] = self.morphology_loss(pred_fine, target)
        losses["einthoven"] = self.einthoven_loss(pred_fine)
        losses["sparse"] = self.sparse_loss(amplitude)
        losses["temporal"] = self.temporal_consistency_loss(pred_fine)
        total = (
            self.w_recon * losses["recon"]
            + self.w_coarse * losses["coarse"]
            + self.w_freq * losses["freq"]
            + self.w_morph * losses["morph"]
            + self.w_einthoven * losses["einthoven"]
            + self.w_sparse * losses["sparse"]
            + self.w_temporal * losses["temporal"]
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
    
    model = CDGS(d_model=256, n_gaussians=64, use_metadata=True)
    
    print(f"CDGS Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    pred_fine, pred_coarse, extras = model(x, age, sex)
    print(f"pred_fine:   {pred_fine.shape}")
    print(f"pred_coarse: {pred_coarse.shape}")
    print("에러 없이 정상적으로 4D SH 렌더링 완료!")