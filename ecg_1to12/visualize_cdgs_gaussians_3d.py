#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
학습된 CDGS / CDGS2 / CDGS3 체크포인트로 가우시안의 단위구면 상 3D 궤적 p̂(t)와 시간축 σ(t)를 시각화합니다.

의미:
  - p̂(t): 렌더러와 동일하게 p_t = p0 + p_vel·(t−0), t∈[0,1] 정규화 후 단위벡터.
  - σ(t): softplus(σ0 + σ_vel·(t−μ)) + ε — **시간 폭**(μ 주변), 공간 3D 공분산이 아님.
  - L_k(t): 12개 리드(카메라) 단위축 — 사지 6 + 흉부 6, 메타데이터 시 흉부 오프셋·호흡 진동으로
    시간에 따라 움직임(GaussianRenderer와 동일 식). PNG 오른쪽= t=0 대비 각 변화(deg), Plotly 점선=궤적.

로컬 (프로젝트 루트):
  python ecg_1to12/visualize_cdgs_gaussians_3d.py \\
      --checkpoint outputs/cdg_2/best.pt --model cdg_2 \\
      --data_dir ./data/ptb-xl --out_path ./viz_gaussians_3d.png

  HTML(회전): --html ./viz_gaussians_3d.html  (plotly 필요)

Docker — **바로 3D 인터랙티브까지** (권장: 스크립트 한 번)
  프로젝트 루트에서:
    chmod +x ecg_1to12/run_docker_gaussians_3d.sh   # 최초 1회
    ./ecg_1to12/run_docker_gaussians_3d.sh
  체크포인트/모델만 바꿀 때:
    CHECKPOINT=outputs/cdg_3/best.pt MODEL=cdg_3 ./ecg_1to12/run_docker_gaussians_3d.sh
  학습 때 direct 옵션 썼으면:
    EXTRA_ARGS="--cdgs_direct_lead_sum" ./ecg_1to12/run_docker_gaussians_3d.sh
  GPU 번호·PTBXL 경로는 GPU_DEVICE, PTBXL_HOST (run_docker_train_nohup.sh 와 동일).

  생성된 viz_gaussians_3d.html 은 레포 루트에 생김 (= 호스트에서 그대로 경로). 브라우저로 열면 회전·확대 가능.
  Linux: xdg-open ./viz_gaussians_3d.html   Windows: 파일 탐색기에서 더블클릭.

  Plotly HTML **보는 법**
  - **아래 슬라이더**: 시간 샘플 t 를 옮기면, 그 시점까지의 **p̂ 누적 궤적** + **현재 t의 12 리드 축** 이 갱신됨.
  - **▶ 재생 / ⏮ 처음**: 자동 재생 또는 t=0 으로 되돌리기.
  - **드래그**: 3D 회전. **스크롤**: 확대/축소.
  - **범례**: 한 번 클릭=숨김/표시, 더블클릭=한 레이어만.
  - 전 구간 궤적을 한 번에 보고 싶으면 --no_plotly_time_slider
  - 구 끄기: --no_plotly_sphere

Docker — **한 줄 복사** (PNG+HTML 동시, 백슬래시 줄 이어쓰기 불필요)
  device / PTBXL 경로 / checkpoint / --model 만 학습과 맞게 수정.
  학습 때 --cdgs_direct_* 썼으면 python ... 인자 끝에 동일 플래그 추가.

  docker run --rm --gpus '"device=6"' -v ~/project_jyu/physionet.org/files/ptb-xl/1.0.3:/workspace/data/ptb-xl -v "$(pwd)":/workspace -w /workspace pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime bash -c "pip install -q wfdb pandas scipy scikit-learn tqdm matplotlib plotly 'numpy<2' && python ecg_1to12/visualize_cdgs_gaussians_3d.py --checkpoint outputs/cdg_12/best.pt --model cdg_12 --data_dir ./data/ptb-xl --out_path ./viz_gaussians_12new.png --html ./viz_gaussians_12new.html"

  PNG만 필요하면 plotly 빼고 --html 인자 제거한 명령으로 실행.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wfdb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class PTBXLReconstructionDataset(Dataset):
    def __init__(self, data_dir: Path, record_paths: List[str], input_idx: int):
        self.data_dir = data_dir
        self.record_paths = record_paths
        self.input_idx = input_idx

    def __len__(self) -> int:
        return len(self.record_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.record_paths[idx]
        signal, _ = wfdb.rdsamp(str(self.data_dir / rec))
        signal = signal.astype(np.float32)
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True) + 1e-6
        signal = (signal - mean) / std
        signal = signal.T
        x = signal[self.input_idx : self.input_idx + 1, :]
        y = signal
        age = torch.tensor(0.5, dtype=torch.float32)
        sex = torch.tensor(1, dtype=torch.long)
        return torch.from_numpy(x), torch.from_numpy(y), age, sex


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_val_records(args: argparse.Namespace) -> List[str]:
    data_dir = Path(args.data_dir)
    csv_path = data_dir / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    file_col = "filename_hr" if args.use_hr else "filename_lr"
    rng = np.random.default_rng(args.seed)
    if "strat_fold" in df.columns:
        val_folds = {int(x.strip()) for x in args.val_folds.split(",") if x.strip()}
        val_df = df[df["strat_fold"].isin(val_folds)]
        val_records = val_df[file_col].tolist()
        rng.shuffle(val_records)
    else:
        records = df[file_col].tolist()
        rng.shuffle(records)
        _, val_records = train_test_split(records, test_size=args.val_ratio, random_state=args.seed)
    return val_records


def compute_p_hat_sigma(
    mu: torch.Tensor,
    sigma0: torch.Tensor,
    sigma_vel: torch.Tensor,
    p0: torch.Tensor,
    p_vel: Optional[torch.Tensor],
    T: int,
    *,
    interior_ball: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mu, sigma0, sigma_vel: (B, N)
    p0: (B, N, 3)
    p_vel: (B, N, 3) or None
    Returns:
      p_hat: (B, N, 3, T) — interior_ball=False 이면 단위구 위 방향, True 이면 cdg_3와 동일 단위구 *내부* 점
      sigma_t: (B, N, T) temporal width
      t_grid: (T,) in [0,1]
    """
    device = mu.device
    dtype = mu.dtype
    B, N = mu.shape
    t = torch.linspace(0, 1, T, device=device, dtype=dtype).view(1, 1, T)
    dt = t - mu.unsqueeze(-1)
    sigma_t = F.softplus(sigma0.unsqueeze(-1) + sigma_vel.unsqueeze(-1) * dt) + 1e-3
    if p_vel is None:
        # cdg_4 이전(고정 p0) 호환: constant over time
        p_hat = p0.unsqueeze(-1).expand(-1, -1, -1, T)
    elif interior_ball:
        from models.cdg_3 import dipole_position_in_unit_ball  # noqa: E402

        p_hat = dipole_position_in_unit_ball(p0, p_vel, dt)
    else:
        p_t = p0.unsqueeze(-1) + p_vel.unsqueeze(-1) * dt.unsqueeze(2)
        p_hat = F.normalize(p_t, dim=2, eps=1e-6)
    return p_hat, sigma_t, t.view(-1)


def compute_p_cellbox_time(
    mu: torch.Tensor,
    sigma0: torch.Tensor,
    sigma_vel: torch.Tensor,
    cell_logits: torch.Tensor,
    delta_code: torch.Tensor,
    delta_mlp,
    codebook: torch.Tensor,
    pos_temp: torch.Tensor,
    hard_positions: bool,
    T: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    cdg_4: 셀 하나를 선택(고정)하고, 셀 내부에서 delta(t)로 연속 이동.
    Returns p_pos: (B,N,3,T), sigma_t: (B,N,T), t_grid: (T,)
    """
    device = mu.device
    dtype = mu.dtype
    t = torch.linspace(0, 1, T, device=device, dtype=dtype).view(1, 1, T)
    dt = t - mu.unsqueeze(-1)
    sigma_t = F.softplus(sigma0.unsqueeze(-1) + sigma_vel.unsqueeze(-1) * dt) + 1e-3

    probs = F.softmax(cell_logits / (pos_temp.to(dtype=dtype) + 1e-6), dim=-1)  # (B,N,K)
    # cdg_4는 1:1 할당을 Sinkhorn(ST)로 만들기 때문에, 여기서도 같은 방식으로 center를 재구성
    try:
        from models.cdg_4 import sinkhorn as _sinkhorn, greedy_unique_argmax as _greedy  # noqa: E402
    except Exception:
        _sinkhorn, _greedy = None, None
    if _sinkhorn is not None and probs.shape[-1] == probs.shape[1]:
        w_soft = _sinkhorn((cell_logits / (pos_temp.to(dtype=dtype) + 1e-6)), n_iters=30)
        if hard_positions and _greedy is not None:
            w_hard = _greedy(w_soft)
            w = w_hard + (w_soft - w_soft.detach())
        else:
            w = w_soft
    else:
        if hard_positions:
            idx = probs.argmax(dim=-1, keepdim=True)
            w_hard = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
            w = w_hard + (probs - probs.detach())
        else:
            w = probs
    center = torch.einsum("bnk,kc->bnc", w, codebook.to(dtype=dtype))  # (B,N,3)

    cell_half = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 7.0], device=device, dtype=dtype).view(1, 1, 3, 1)
    dt_feat = dt.unsqueeze(2).permute(0, 1, 3, 2)  # (B,N,T,1)
    code_feat = delta_code.unsqueeze(2).expand(-1, -1, T, -1)  # (B,N,T,C)
    mlp_in = torch.cat([code_feat, dt_feat], dim=-1)  # (B,N,T,C+1)
    d_raw = delta_mlp(mlp_in)  # (B,N,T,3)
    delta_t = torch.tanh(d_raw).permute(0, 1, 3, 2) * cell_half
    p_pos = center.unsqueeze(-1) + delta_t
    return p_pos, sigma_t, t.view(-1)


def compute_p_cellbox_time_legacy_delta(
    mu: torch.Tensor,
    sigma0: torch.Tensor,
    sigma_vel: torch.Tensor,
    cell_logits: torch.Tensor,
    delta0: torch.Tensor,
    delta_vel: torch.Tensor,
    codebook: torch.Tensor,
    pos_temp: torch.Tensor,
    hard_positions: bool,
    T: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    레거시 cdg_4(128): 셀 하나 선택(soft/hard) + delta(t)=tanh(delta0+delta_vel*(t-μ))*cell_half
    """
    device = mu.device
    dtype = mu.dtype
    t = torch.linspace(0, 1, T, device=device, dtype=dtype).view(1, 1, T)
    dt = t - mu.unsqueeze(-1)
    sigma_t = F.softplus(sigma0.unsqueeze(-1) + sigma_vel.unsqueeze(-1) * dt) + 1e-3

    probs = F.softmax(cell_logits / (pos_temp.to(dtype=dtype) + 1e-6), dim=-1)  # (B,N,K)
    if hard_positions:
        idx = probs.argmax(dim=-1, keepdim=True)
        w_hard = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
        w = w_hard + (probs - probs.detach())
    else:
        w = probs
    center = torch.einsum("bnk,kc->bnc", w, codebook.to(dtype=dtype))  # (B,N,3)

    # legacy 셀 반폭 (4×4×8)
    cell_half = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 7.0], device=device, dtype=dtype).view(1, 1, 3, 1)
    delta_t = torch.tanh(delta0.unsqueeze(-1) + delta_vel.unsqueeze(-1) * dt.unsqueeze(2)) * cell_half
    p_pos = center.unsqueeze(-1) + delta_t
    return p_pos, sigma_t, t.view(-1)


def compute_lead_vectors_time(model: torch.nn.Module, age: torch.Tensor, sex: torch.Tensor, T: int) -> torch.Tensor:
    """
    GaussianRenderer.forward 와 동일하게 12개 리드(카메라) 단위축 L_k(t).
    Returns (B, 12, 3, T). 사지 6 + 흉부 6, 메타데이터 시 chest_offset·호흡 진동 반영.
    """
    renderer = model.renderer
    B = age.shape[0]
    device = age.device
    dtype = next(renderer.parameters()).dtype
    t = torch.linspace(0, 1, T, device=device, dtype=dtype).view(1, 1, T)

    # cdg_8 등 완전 물리 렌더러는 12개 방향 벡터 대신 9개 좌표점 전극을 씀
    if not hasattr(renderer, "limb_vectors"):
        E = renderer.electrode_anchor + torch.tanh(renderer.electrode_offset) * 0.1
        lead_vecs_t = E.unsqueeze(0).unsqueeze(-1).expand(B, 9, 3, T)
        return lead_vecs_t

    limb = renderer.limb_vectors()
    chest = F.normalize(renderer.chest_vectors, dim=-1)
    chest_offset, resp_freq, resp_vec = None, None, None
    use_meta = bool(getattr(model, "use_metadata", False)) and getattr(model, "meta_conditioner", None) is not None
    if use_meta:
        chest_offset, resp_freq, resp_vec = model.meta_conditioner(age, sex)

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

    # cdg_6: learnable camera motion (if present)
    if bool(getattr(renderer, "enable_camera_motion", False)) and hasattr(renderer, "cam_freq_raw"):
        freq = F.softplus(getattr(renderer, "cam_freq_raw")) + 1e-6
        phase = 2 * math.pi * freq * t  # (1,1,T)
        amp_sin = getattr(renderer, "cam_amp_sin", None)
        amp_cos = getattr(renderer, "cam_amp_cos", None)
        if amp_sin is not None and amp_cos is not None:
            motion = (
                amp_sin.to(device=device, dtype=dtype).view(1, 12, 3, 1) * torch.sin(phase.view(1, 1, 1, T))
                + amp_cos.to(device=device, dtype=dtype).view(1, 12, 3, 1) * torch.cos(phase.view(1, 1, 1, T))
            )
            lead_vecs_t = F.normalize(lead_vecs_t + motion, dim=2)
    return lead_vecs_t


def lead_angle_drift_deg(lead_vecs: np.ndarray) -> np.ndarray:
    """lead_vecs: (12, 3, T) — 각 리드 축이 t=0 대비 얼마나 도는지 (deg), shape (12, T)."""
    L0 = lead_vecs[:, :, :1]
    dot = np.sum(lead_vecs * L0, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return (np.arccos(dot) * 180.0 / math.pi).astype(np.float32)


def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=1e-6)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (...,3,3) columns


def compute_cov3_from_scales_rot6d(scales: torch.Tensor, rot6d: torch.Tensor) -> torch.Tensor:
    """
    scales: (B,N,3) positive
    rot6d:  (B,N,6)
    returns cov3: (B,N,3,3) = R diag(s^2) R^T
    """
    R = _rotation_6d_to_matrix(rot6d)
    s2 = (scales ** 2).clamp_min(1e-8)
    return R @ torch.diag_embed(s2) @ R.transpose(-1, -2)


def _offdiag_values(mat: torch.Tensor) -> torch.Tensor:
    n = mat.size(0)
    if n <= 1:
        return mat.new_zeros(1)
    mask = ~torch.eye(n, dtype=torch.bool, device=mat.device)
    return mat[mask]


def compute_over_sync_metrics(
    sigma_t: torch.Tensor,
    amplitude: Optional[torch.Tensor] = None,
    top_k: int = 0,
    gaussian_dirs_t: Optional[torch.Tensor] = None,
) -> dict:
    """
    과동기화 진단용 요약 지표.
    sigma_t: [B, N, T]
    amplitude: [B, N] (선택)
    gaussian_dirs_t: [B, N, 3, T] (선택)
    """
    env = sigma_t[0].float()  # [N, T]
    n_slots = env.size(0)

    if amplitude is not None and top_k > 0 and top_k < n_slots:
        idx = torch.topk(amplitude[0], k=top_k).indices
    else:
        idx = torch.arange(n_slots, device=env.device)

    env = env[idx]  # [K, T]
    k_slots = env.size(0)

    # 슬롯별 시간 신호 상관: 평균 제거 후 cosine
    env_centered = env - env.mean(dim=-1, keepdim=True)
    env_norm = F.normalize(env_centered, dim=-1, eps=1e-6)
    env_sim = env_norm @ env_norm.transpose(0, 1)  # [K, K]
    off = _offdiag_values(env_sim)

    peak_idx = env.argmax(dim=-1).float()
    peak_spread_norm = (peak_idx.std(unbiased=False) / max(env.size(-1) - 1, 1)).item()

    # 유효 rank(표현 다양성): 클수록 슬롯 분화가 큼
    svals = torch.linalg.svdvals(env_norm)
    power = svals.pow(2)
    p = power / (power.sum() + 1e-8)
    eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum()).item()

    metrics = {
        "slots_used": int(k_slots),
        "env_cos_mean": float(off.mean().item()),
        "env_cos_median": float(off.median().item()),
        "env_cos_p90": float(torch.quantile(off, 0.90).item()),
        "env_frac_gt_08": float((off > 0.8).float().mean().item()),
        "peak_spread_norm": float(peak_spread_norm),
        "effective_rank": float(eff_rank),
    }

    if gaussian_dirs_t is not None:
        dirs = gaussian_dirs_t[0, idx].float()  # [K, 3, T]
        dirs = F.normalize(dirs, dim=1, eps=1e-6)

        # time별 평균 방향 벡터 길이: 1에 가까울수록 방향 정렬 강함
        r_t = dirs.mean(dim=0).norm(dim=0)  # [T]

        # time별 pairwise cosine 평균
        ts = torch.linspace(
            0,
            dirs.size(-1) - 1,
            steps=min(128, dirs.size(-1)),
            device=dirs.device,
        ).long().unique()
        pair_means = []
        for ti in ts:
            d_t = dirs[:, :, ti]  # [K, 3]
            sim_t = d_t @ d_t.transpose(0, 1)
            pair_means.append(_offdiag_values(sim_t).mean())

        if pair_means:
            dir_pair_mean = torch.stack(pair_means).mean().item()
        else:
            dir_pair_mean = 0.0

        metrics.update({
            "dir_resultant_mean": float(r_t.mean().item()),
            "dir_resultant_p90": float(torch.quantile(r_t, 0.90).item()),
            "dir_pair_cos_mean": float(dir_pair_mean),
        })

    return metrics

_LEAD_LINE_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
)


def _display_order(amp: np.ndarray, sig: np.ndarray, top_k: int, sort_by: str) -> np.ndarray:
    if sort_by == "peak_time":
        peak_idx = sig.argmax(axis=1)
        # peak time 오름차순, 동일 peak에서는 amplitude 큰 슬롯 우선
        order = np.lexsort((-amp, peak_idx))
    else:
        order = np.argsort(-amp)
    if top_k > 0:
        order = order[:top_k]
    return order


def plot_matplotlib(
    p_hat: np.ndarray,
    sigma_t: np.ndarray,
    amplitude: np.ndarray,
    mu: np.ndarray,
    out_path: Path,
    top_k: int,
    sphere_alpha: float,
    lead_vecs_bt: Optional[np.ndarray] = None,
    interior_ball_trajectory: bool = False,
    jitter_scale: float = 0.0,
    cov3_bt: Optional[np.ndarray] = None,
    gaussian_dirs_bt: Optional[np.ndarray] = None,
    direction_scale: float = 0.08,
    sort_by: str = "amp",
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # b=0
    p = p_hat[0]  # (N, 3, T)
    sig = sigma_t[0]  # (N, T)
    amp = amplitude[0]  # (N,)
    mu_n = mu[0]

    N = p.shape[0]
    T = p.shape[2]
    order = _display_order(amp, sig, top_k, sort_by)

    # 겹침 완화: 같은(또는 매우 가까운) 위치에 많은 슬롯이 포개지면 점이 사라진 것처럼 보임.
    # 각 슬롯 n마다 아주 작은 3D 오프셋(jitter)을 trajectory 전체에 더해 시각적으로 분리.
    if float(jitter_scale) > 0:
        rng = np.random.default_rng(0)
        jit = rng.normal(size=(N, 3)).astype(np.float32)
        jit /= (np.linalg.norm(jit, axis=1, keepdims=True) + 1e-8)
        # amp 큰 것일수록 더 잘 보이게 약간 키움
        a = (amp / (amp.max() + 1e-6)).astype(np.float32)
        jit = jit * (float(jitter_scale) * (0.35 + 0.65 * a)[:, None])
        p = p + jit[:, :, None]

    fig = plt.figure(figsize=(18, 6))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    _title_3d = (
        "p(t) in unit ball (tanh(r)·dir, cdg_3)"
        if interior_ball_trajectory
        else "p̂(t) on unit sphere (Gaussians)"
    )
    if gaussian_dirs_bt is not None:
        _title_3d += " + gaussian dipole direction d(t)"
    if lead_vecs_bt is not None:
        _title_3d += " + L_k(t) dashed (12 lead axes)"
    ax3d.set_title(_title_3d)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax3d.plot_wireframe(xs, ys, zs, color="gray", alpha=sphere_alpha, linewidth=0.3)

    try:
        cmap = plt.colormaps["tab20"]
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20", 20)
    for rank, n in enumerate(order):
        traj = p[n].T  # (T, 3)
        c = cmap(rank % 20)
        ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=c, alpha=0.85, linewidth=1.2, label=None)
        mu_idx = int(np.clip(mu_n[n] * (T - 1), 0, T - 1))
        ax3d.scatter(
            [traj[mu_idx, 0]],
            [traj[mu_idx, 1]],
            [traj[mu_idx, 2]],
            color=[c],
            s=25 + 80 * float(amp[n] / (amp.max() + 1e-6)),
            marker="o",
        )

        if gaussian_dirs_bt is not None:
            d_nt = gaussian_dirs_bt[int(n)]  # (3, T)
            s_cnt = min(12, T)
            t_idx = np.unique(np.linspace(0, T - 1, num=s_cnt).astype(np.int64))
            for ti in t_idx:
                p0 = p[int(n), :, int(ti)]
                d0 = d_nt[:, int(ti)]
                p1 = p0 + float(direction_scale) * d0
                ax3d.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]],
                    color=c,
                    alpha=0.30,
                    linewidth=0.8,
                    solid_capstyle="round",
                )

            # μ 시점 방향은 조금 더 굵게 표시
            p0 = p[int(n), :, mu_idx]
            d0 = d_nt[:, mu_idx]
            p1 = p0 + float(direction_scale) * d0
            ax3d.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=c,
                alpha=0.9,
                linewidth=1.8,
                solid_capstyle="round",
            )

    ax3d.set_box_aspect((1, 1, 1))

    # (optional) per-gaussian spatial covariance ellipsoids at t=0
    if cov3_bt is not None:
        cov3 = cov3_bt[0]  # (N,3,3)
        t0 = 0
        
        # Base sphere for ellipsoid surface
        u_ell = np.linspace(0, 2 * np.pi, 16)
        v_ell = np.linspace(0, np.pi, 8)
        base_x = np.outer(np.cos(u_ell), np.sin(v_ell))
        base_y = np.outer(np.sin(u_ell), np.sin(v_ell))
        base_z = np.outer(np.ones(np.size(u_ell)), np.cos(v_ell))
        base_pts = np.stack([base_x, base_y, base_z], axis=-1)  # (16, 8, 3)
        
        for rank, n in enumerate(order):
            C = cov3[int(n)]
            c = cmap(rank % 20)
            w, V = np.linalg.eigh(C.astype(np.float64))
            w = np.maximum(w, 1e-10)
            
            # transform base sphere
            scale_mat = np.diag(np.sqrt(w)) * 0.5  # 0.5 scale for better visual fit
            transform = V @ scale_mat
            
            # base_pts: (..., 3), transform: (3, 3) => pts @ transform.T
            ell_pts = np.einsum('ijk,lk->ijl', base_pts, transform) 
            
            center = p[int(n), :, t0]
            ell_x = ell_pts[..., 0] + center[0]
            ell_y = ell_pts[..., 1] + center[1]
            ell_z = ell_pts[..., 2] + center[2]
            
            ax3d.plot_surface(ell_x, ell_y, ell_z, color=c, alpha=0.1, shade=True, linewidth=0)

    if lead_vecs_bt is not None:
        L = lead_vecs_bt
        for k in range(L.shape[0]):
            traj_l = L[k].T
            ax3d.plot(
                traj_l[:, 0],
                traj_l[:, 1],
                traj_l[:, 2],
                color=_LEAD_LINE_COLORS[k % len(_LEAD_LINE_COLORS)],
                alpha=0.45,
                linewidth=0.9,
                linestyle=(0, (4, 2)),
                label=None,
            )

    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(sig, aspect="auto", cmap="magma", interpolation="nearest")
    ax2.set_xlabel("time index t (0…T−1)")
    ax2.set_ylabel("Gaussian slot n")
    ax2.set_title("σ(t): temporal width per slot (softplus(σ0+σ_vel·Δt)+ε)")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(1, 3, 3)
    if lead_vecs_bt is not None:
        drift = lead_angle_drift_deg(lead_vecs_bt)
        im2 = ax3.imshow(drift, aspect="auto", cmap="viridis", interpolation="nearest")
        ax3.set_yticks(range(L.shape[0]))
        labels = ["RA", "LA", "LL", "V1", "V2", "V3", "V4", "V5", "V6"] if L.shape[0] == 9 else _LEADS
        ax3.set_yticklabels(labels, fontsize=7)
        ax3.set_xlabel("time index t")
        ax3.set_title("Electrode Position / Lead Drift (deg)")
        plt.colorbar(im2, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _format_amp(a: float) -> str:
    if abs(a) < 1e-2:
        return f"{a:.2e}"
    return f"{a:.4g}"


def _plotly_sphere_trace(go, sphere_opacity: float):
    theta = np.linspace(0.0, np.pi, 22)
    phi = np.linspace(0.0, 2 * np.pi, 44)
    Th, Ph = np.meshgrid(theta, phi, indexing="ij")
    xs = np.sin(Th) * np.cos(Ph)
    ys = np.sin(Th) * np.sin(Ph)
    zs = np.cos(Th)
    return go.Surface(
        x=xs,
        y=ys,
        z=zs,
        opacity=float(sphere_opacity),
        showscale=False,
        surfacecolor=zs,
        cmin=-1,
        cmax=1,
        colorscale=[[0.0, "rgba(160,160,170,0.55)"], [1.0, "rgba(160,160,170,0.55)"]],
        name="단위구 (범례 클릭으로 끄기)",
        hoverinfo="skip",
        lighting=dict(ambient=0.95, diffuse=0.35, specular=0.2),
    )


def _slider_frame_traces(
    p: np.ndarray,
    order: np.ndarray,
    amp: np.ndarray,
    ti: int,
    line_colors: Tuple[str, ...],
    lead_vecs: Optional[np.ndarray],
    show_lead_cameras: bool,
    jitter_scale: float = 0.0,
    sigma_t: Optional[np.ndarray] = None,
    gaussian_dirs: Optional[np.ndarray] = None,
    direction_scale: float = 0.08,
) -> list:
    """시간 인덱스 ti 까지의 궤적 + 현재 시점 리드 축 점."""
    import plotly.graph_objects as go

    out = []
    T = p.shape[2]
    ti = int(np.clip(ti, 0, T - 1))
    if float(jitter_scale) > 0:
        rng = np.random.default_rng(0)
        jit = rng.normal(size=(p.shape[0], 3)).astype(np.float32)
        jit /= (np.linalg.norm(jit, axis=1, keepdims=True) + 1e-8)
        a = (amp / (amp.max() + 1e-6)).astype(np.float32)
        jit = jit * (float(jitter_scale) * (0.35 + 0.65 * a)[:, None])
    for rank, n in enumerate(order):
        traj = p[n].T[: ti + 1]
        if float(jitter_scale) > 0:
            traj = traj + jit[n][None, :]
        c = line_colors[rank % len(line_colors)]
        Ln = traj.shape[0]
        if Ln < 1:
            traj = p[n].T[:1]
            Ln = 1
            
        # 파동(envelope_t) 값이 들어왔다면 현재 시간에 맞게 크기를 펄스(맥박)처럼 변경!
        current_size = 14
        if sigma_t is not None:
            act = float(sigma_t[n, ti]) # 0~1 사이의 파동 값
            current_size = 5 + act * 35 # 최소 5, 최대 40까지 부풀어오름
            
        ms = [0] * (Ln - 1) + [current_size] if Ln > 1 else [current_size]
        out.append(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines+markers",
                line=dict(width=3, color=c),
                marker=dict(size=ms, color=c, opacity=1.0),
                name=f"n={n} amp={_format_amp(float(amp[n]))}",
                legendgroup=str(n),
                showlegend=(rank < 8),
            )
        )

        if gaussian_dirs is not None:
            center = traj[-1]
            dvec = gaussian_dirs[n, :, ti]
            tip = center + float(direction_scale) * dvec
            out.append(
                go.Scatter3d(
                    x=[center[0], tip[0]],
                    y=[center[1], tip[1]],
                    z=[center[2], tip[2]],
                    mode="lines+markers",
                    line=dict(width=4, color=c),
                    marker=dict(size=[0, 4], color=c, opacity=1.0),
                    name=f"dir n={n}",
                    legendgroup=f"dir-{n}",
                    showlegend=False,
                )
            )
    if show_lead_cameras and lead_vecs is not None:
        P = lead_vecs[:, :, ti]
        labels = ["RA", "LA", "LL", "V1", "V2", "V3", "V4", "V5", "V6"] if P.shape[0] == 9 else list(_LEADS)
        out.append(
            go.Scatter3d(
                x=P[:, 0],
                y=P[:, 1],
                z=P[:, 2],
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont=dict(size=12, color="black"),
                marker=dict(size=8, color=list(_LEAD_LINE_COLORS), opacity=0.95),
                name="리드 축 L_k (현재 t)",
                legendgroup="lead_cameras",
            )
        )
    return out


def plot_plotly(
    p_hat: np.ndarray,
    sigma_t: np.ndarray,
    amplitude: np.ndarray,
    mu: np.ndarray,
    out_html: Path,
    top_k: int,
    *,
    show_unit_sphere: bool = True,
    sphere_opacity: float = 0.14,
    lead_vecs: Optional[np.ndarray] = None,
    show_lead_cameras: bool = True,
    plotly_time_slider: bool = True,
    max_time_slider_frames: int = 160,
    interior_ball_trajectory: bool = False,
    jitter_scale: float = 0.0,
    cov3_bt: Optional[np.ndarray] = None,
    gaussian_dirs_bt: Optional[np.ndarray] = None,
    direction_scale: float = 0.08,
    sort_by: str = "amp",
) -> None:
    import plotly.graph_objects as go

    line_colors = (
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    )
    p = p_hat[0]
    sig = sigma_t[0]
    amp = amplitude[0]
    mu_n = mu[0]
    N, _, T = p.shape
    order = _display_order(amp, sig, top_k, sort_by)

    # Covariance data extraction for Plotly Mesh3d
    cov3_ellipsoids = []
    if cov3_bt is not None:
        from scipy.spatial import ConvexHull
        cov3 = cov3_bt[0]
        
        # Icosahedron vertices for a low-poly ellipsoid
        phi = (1 + math.sqrt(5)) / 2
        vs = np.array([
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ], dtype=np.float32)
        vs /= np.linalg.norm(vs, axis=1, keepdims=True)
        hull = ConvexHull(vs)
        i, j, k = hull.simplices.T
        
        for n in range(N):
            C = cov3[n]
            w, V = np.linalg.eigh(C.astype(np.float64))
            w = np.maximum(w, 1e-10)
            transform = V @ (np.diag(np.sqrt(w)) * 0.5)
            pts = vs @ transform.T
            cov3_ellipsoids.append({'pts': pts, 'i': i, 'j': j, 'k': k})

    scene_kw = dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
        aspectmode="cube",
        xaxis=dict(range=[-1.15, 1.15], backgroundcolor="rgb(250,250,252)"),
        yaxis=dict(range=[-1.15, 1.15], backgroundcolor="rgb(250,250,252)"),
        zaxis=dict(range=[-1.15, 1.15], backgroundcolor="rgb(250,250,252)"),
    )

    if plotly_time_slider:
        cap = max(2, int(max_time_slider_frames))
        k = min(cap, T)
        if k >= T:
            time_indices = np.arange(T, dtype=int)
        else:
            time_indices = np.unique(np.linspace(0, T - 1, num=k).astype(np.int64))

        trace_off = 1 if show_unit_sphere else 0
        data0: list = []
        if show_unit_sphere:
            data0.append(_plotly_sphere_trace(go, sphere_opacity))
        data0.extend(
            _slider_frame_traces(
                p,
                order,
                amp,
                int(time_indices[0]),
                line_colors,
                lead_vecs,
                show_lead_cameras,
                jitter_scale=jitter_scale,
                sigma_t=sig,
                gaussian_dirs=gaussian_dirs_bt,
                direction_scale=direction_scale,
            )
        )
        if cov3_ellipsoids:
            for rank, n in enumerate(order):
                center = p[n, :, 0]
                ell = cov3_ellipsoids[n]
                c = line_colors[rank % len(line_colors)]
                data0.append(go.Mesh3d(
                    x=ell['pts'][:, 0] + center[0],
                    y=ell['pts'][:, 1] + center[1],
                    z=ell['pts'][:, 2] + center[2],
                    i=ell['i'], j=ell['j'], k=ell['k'],
                    color=c, opacity=0.1, name=f"cov3 n={n}", hoverinfo='skip', showlegend=False
                ))

        frames = []
        for ti in time_indices:
            fr_tr = _slider_frame_traces(
                p,
                order,
                amp,
                int(ti),
                line_colors,
                lead_vecs,
                show_lead_cameras,
                jitter_scale=jitter_scale,
                sigma_t=sig,
                gaussian_dirs=gaussian_dirs_bt,
                direction_scale=direction_scale,
            )
            n_anim = len(fr_tr)
            frames.append(
                go.Frame(
                    data=fr_tr,
                    traces=list(range(trace_off, trace_off + n_anim)),
                    name=f"t{int(ti)}",
                )
            )

        fig = go.Figure(data=data0, frames=frames)
        n_steps = len(time_indices)
        slider_steps = [
            {
                "args": [
                    [frames[j].name],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": f"{int(time_indices[j])}",
                "method": "animate",
            }
            for j in range(n_steps)
        ]
        _sl_title = (
            "시간 슬라이더: t — 구 내부 p(t) 누적 + L_k (cdg_3)"
            if interior_ball_trajectory
            else "시간 슬라이더: 샘플 인덱스 t — p̂ 누적 궤적 + 현재 L_k (점/라벨)"
        )
        if gaussian_dirs_bt is not None:
            _sl_title += " + Gaussian dir(t)"
        fig.update_layout(
            title=_sl_title,
            scene=scene_kw,
            margin=dict(l=0, r=0, t=48, b=120),
            legend=dict(font=dict(size=9), tracegroupgap=0),
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "left",
                    "showactive": False,
                    "x": 0.05,
                    "y": 0.02,
                    "xanchor": "left",
                    "yanchor": "top",
                    "buttons": [
                        {
                            "label": "▶ 재생",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 35, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "⏮ 처음",
                            "method": "animate",
                            "args": [
                                [frames[0].name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "prefix": "시간 샘플 t = ",
                        "visible": True,
                        "xanchor": "left",
                    },
                    "pad": {"b": 10, "t": 40},
                    "len": 0.9,
                    "x": 0.05,
                    "y": 0,
                    "steps": slider_steps,
                }
            ],
        )
    else:
        traces: list = []
        if show_unit_sphere:
            traces.append(_plotly_sphere_trace(go, sphere_opacity))

        for rank, n in enumerate(order):
            traj = p[n].T
            mu_idx = int(np.clip(mu_n[n] * (T - 1), 0, T - 1))
            w = 1.0 + 4.0 * (sig[n] / (sig[n].max() + 1e-6))
            c = line_colors[rank % len(line_colors)]
            traces.append(
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2],
                    mode="lines",
                    line=dict(width=3, color=c),
                    name=f"n={n} amp={_format_amp(float(amp[n]))}",
                    legendgroup=str(n),
                )
            )
            traces.append(
                go.Scatter3d(
                    x=[traj[mu_idx, 0]],
                    y=[traj[mu_idx, 1]],
                    z=[traj[mu_idx, 2]],
                    mode="markers",
                    marker=dict(size=6 + 10 * w[mu_idx], opacity=0.9, color=c),
                    name=f"μ n={n}",
                    legendgroup=str(n),
                    showlegend=False,
                )
            )

            if gaussian_dirs_bt is not None:
                d_nt = gaussian_dirs_bt[int(n)]
                s_cnt = min(14, T)
                t_idx = np.unique(np.linspace(0, T - 1, num=s_cnt).astype(np.int64))
                x_dir, y_dir, z_dir = [], [], []
                for ti in t_idx:
                    p0 = p[int(n), :, int(ti)]
                    d0 = d_nt[:, int(ti)]
                    p1 = p0 + float(direction_scale) * d0
                    x_dir.extend([p0[0], p1[0], None])
                    y_dir.extend([p0[1], p1[1], None])
                    z_dir.extend([p0[2], p1[2], None])
                traces.append(
                    go.Scatter3d(
                        x=x_dir,
                        y=y_dir,
                        z=z_dir,
                        mode="lines",
                        line=dict(width=2, color=c),
                        name=f"dir n={n}",
                        legendgroup=f"dir-{n}",
                        showlegend=False,
                    )
                )
            if cov3_ellipsoids:
                ell = cov3_ellipsoids[n]
                center = traj[0]
                traces.append(go.Mesh3d(
                    x=ell['pts'][:, 0] + center[0],
                    y=ell['pts'][:, 1] + center[1],
                    z=ell['pts'][:, 2] + center[2],
                    i=ell['i'], j=ell['j'], k=ell['k'],
                    color=c, opacity=0.1, name=f"Cov n={n}", hoverinfo='skip', showlegend=False, legendgroup=str(n)
                ))

        if show_lead_cameras and lead_vecs is not None:
            L = lead_vecs
            for k in range(12):
                traj_l = L[k].T
                c = _LEAD_LINE_COLORS[k % 12]
                traces.append(
                    go.Scatter3d(
                        x=traj_l[:, 0],
                        y=traj_l[:, 1],
                        z=traj_l[:, 2],
                        mode="lines",
                        line=dict(width=2, color=c, dash="dash"),
                        name=f"카메라 {_LEADS[k]}",
                        legendgroup="lead_cameras",
                    )
                )

        fig = go.Figure(data=traces)
        if interior_ball_trajectory:
            _ttl = "p(t) in unit ball + L_k(t) (cdg_3)"
        else:
            _ttl = "p̂(t) 가우시안 + 12 리드 카메라 축 L_k(t) (점선)"
        if gaussian_dirs_bt is not None:
            _ttl += " + Gaussian dir(t)"
        if not (show_lead_cameras and lead_vecs is not None):
            _ttl = "p(t) unit ball (cdg_3)" if interior_ball_trajectory else "CDGS Gaussian directions p̂(t)"
            if gaussian_dirs_bt is not None:
                _ttl += " + dir(t)"
        fig.update_layout(
            title=_ttl + " — 드래그 회전 · 범례 클릭으로 레이어 끄기",
            scene=scene_kw,
            margin=dict(l=0, r=0, t=48, b=0),
            legend=dict(font=dict(size=10), tracegroupgap=0),
        )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, config=dict(displayModeBar=True, displaylogo=False), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="cdg_2",
        choices=("cdgs", "cdg_2", "cdg_3", "cdg_4", "cdg_5", "cdg_6", "cdg_7", "cdg_8", "cdg_10", "cdg_11", "cdg_12"),
    )
    parser.add_argument("--data_dir", type=str, default="./data/ptb-xl")
    parser.add_argument("--input_lead", type=str, default="I", choices=_LEADS)
    parser.add_argument("--use_hr", action="store_true")
    parser.add_argument("--val_folds", type=str, default="10")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--batch_index", type=int, default=0, help="DataLoader에서 몇 번째 배치의 첫 샘플(b=0)을 그릴지")
    parser.add_argument("--cdgs_direct_lead_sum", action="store_true")
    parser.add_argument("--cdgs_direct_patch", type=int, default=5)
    parser.add_argument("--cdgs_direct_span", type=float, default=0.55)
    parser.add_argument("--out_path", type=str, default="./viz_gaussians_3d.png")
    parser.add_argument("--html", type=str, default="", help="지정 시 plotly HTML 저장 (pip install plotly)")
    parser.add_argument(
        "--no_plotly_sphere",
        action="store_true",
        help="Plotly HTML에서 단위구 Surface 끄기(궤적만 보고 싶을 때)",
    )
    parser.add_argument(
        "--plotly_sphere_opacity",
        type=float,
        default=0.14,
        help="Plotly 단위구 투명도 (0~1, 기본 0.14)",
    )
    parser.add_argument(
        "--no_plotly_time_slider",
        action="store_true",
        help="Plotly HTML을 전 구간 정적 궤적으로만 저장 (시간 슬라이더 없음)",
    )
    parser.add_argument(
        "--plotly_max_time_frames",
        type=int,
        default=160,
        help="시간 슬라이더 프레임 수 상한 (T가 크면 균등 샘플링)",
    )
    parser.add_argument("--top_k", type=int, default=32, help="진폭 상위 K개 슬롯만 3D에 그림 (0이면 전부)")
    parser.add_argument(
        "--no_lead_cameras",
        action="store_true",
        help="12 리드(카메라) 축 L_k(t) 궤적·각변위 패널 끄기",
    )
    parser.add_argument(
        "--no_gaussian_directions",
        action="store_true",
        help="가우시안 방향 벡터 dir(t) 오버레이 끄기 (cdg_10/cdg_11/cdg_12에서 dipole_dir_t 사용)",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="amp",
        choices=("amp", "peak_time"),
        help="가우시안 표시 순서 (amp: 진폭순, peak_time: envelope peak 시점순)",
    )
    parser.add_argument(
        "--direction_scale",
        type=float,
        default=0.08,
        help="가우시안 방향 벡터 표시 길이 스케일",
    )
    parser.add_argument("--sphere_alpha", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise SystemExit("matplotlib 필요: pip install matplotlib") from e

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ecg_models"))
    from models import CDGS, CDGS2, CDGS3, CDGS4, CDGS5, CDGS6, CDGS7, CDGS8, CDGS10, CDGS11, CDGS12  # noqa: E402

    # 레거시 cdg_4(128) 체크포인트 로더 (cdg_4.py가 64로 바뀐 뒤에도 로드 가능).
    # cdg_4 모듈에 의존하지 않고, 시각화 스크립트 내부에 "최소 구성"을 내장한다.
    #
    # 단, 레거시 ckpt의 encoder 파라미터 이름을 맞추기 위해 MultiScaleEncoder는
    # 현재 cdg_4 모듈의 구현을 그대로 가져와 사용한다(구조는 동일).

    try:
        from models.cdg_4 import MultiScaleEncoder as _LegacyMultiScaleEncoder  # noqa: E402
    except Exception:
        _LegacyMultiScaleEncoder = None

    def _build_codebook_128(device: torch.device, dtype: torch.dtype, *, r_max: float = 0.92) -> torch.Tensor:
        xs = torch.linspace(-1.0, 1.0, 4, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, 4, device=device, dtype=dtype)
        zs = torch.linspace(-1.0, 1.0, 8, device=device, dtype=dtype)
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(-1, 3)  # (128,3)
        nrm = grid.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.minimum(torch.ones_like(nrm), (r_max / nrm))
        return grid * scale

    class _LegacyGaussianPredictor128(torch.nn.Module):
        def __init__(self, d_model: int = 256, n_gaussians: int = 128, n_heads: int = 8):
            super().__init__()
            self.n_gaussians = int(n_gaussians)
            self.sh_dim = 9
            self.codebook_size = 128
            out_dim = 4 + self.sh_dim * 2 + self.codebook_size + 3 + 3  # =156
            _gq = torch.randn(1, self.n_gaussians, d_model)
            self.gaussian_queries = torch.nn.Parameter(torch.nn.functional.normalize(_gq, dim=-1, eps=1e-6))
            self.cross_attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.param_head = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model // 2),
                torch.nn.GELU(),
                torch.nn.Linear(d_model // 2, out_dim),
            )

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
            amplitude = torch.nn.functional.softplus(params[..., 3]) + 1e-6
            sh_base = params[..., 4:13]
            sh_vel = params[..., 13:22]
            cell_logits = params[..., 22 : 22 + 128]
            delta0 = params[..., 22 + 128 : 22 + 128 + 3]
            delta_vel = params[..., 22 + 128 + 3 : 22 + 128 + 6]
            # legacy에는 온도 파라미터가 없었으니 1.0으로 둔다(softmax에만 쓰임)
            pos_temp = mu.new_tensor(1.0)
            return mu, sigma0, sigma_vel, amplitude, sh_base, sh_vel, cell_logits, delta0, delta_vel, pos_temp

    class _CDGS4Legacy128(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_metadata = False
            if _LegacyMultiScaleEncoder is None:
                raise ImportError("legacy cdg_4(128) loader requires MultiScaleEncoder from models.cdg_4")
            self.encoder = _LegacyMultiScaleEncoder(d_model=256, n_layers=4)
            cb = _build_codebook_128(device=torch.device("cpu"), dtype=torch.float32)
            self.register_buffer("pos_codebook", cb)
            self.gaussian_predictor = _LegacyGaussianPredictor128(d_model=256, n_gaussians=128, n_heads=8)

        def forward(self, x, age=None, sex=None):
            _, global_feat = self.encoder(x)
            mu, sigma0, sigma_vel, amplitude, sh_base, sh_vel, cell_logits, delta0, delta_vel, pos_temp = self.gaussian_predictor(
                global_feat
            )
            # coarse/fine은 시각화에서 안 쓰지만, 호출부 호환을 위해 더미 반환
            dummy = x.new_zeros(x.size(0), 12, x.size(2))
            return dummy, dummy, {
                "mu": mu,
                "sigma0": sigma0,
                "sigma_vel": sigma_vel,
                "amplitude": amplitude,
                "cell_logits": cell_logits,
                "delta0": delta0,
                "delta_vel": delta_vel,
                "pos_temp": pos_temp,
            }

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    input_idx = _LEADS.index(args.input_lead)
    val_records = build_val_records(args)
    val_ds = PTBXLReconstructionDataset(data_dir, val_records, input_idx)
    loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    ctor = (
        CDGS
        if args.model == "cdgs"
        else CDGS2
        if args.model == "cdg_2"
        else CDGS3
        if args.model == "cdg_3"
        else CDGS4
        if args.model == "cdg_4"
        else CDGS5
        if args.model == "cdg_5"
        else CDGS6
        if args.model == "cdg_6"
        else CDGS7
        if args.model == "cdg_7"
        else CDGS8
        if args.model == "cdg_8"
        else CDGS10
        if args.model == "cdg_10"
        else CDGS12
        if args.model == "cdg_12"
        else CDGS11
    )
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Auto-detect parameters from checkpoint state_dict (d_model, n_gaussians, n_encoder_layers, etc.)
    d_model = 256
    n_gaussians = 128 if args.model in ("cdg_2", "cdg_3") else 64
    n_encoder_layers = 4
    d_deform_hidden = 64

    ms_ref = ckpt.get("model_state", ckpt)
    if "gaussian_predictor.gaussian_queries" in ms_ref:
        q_shape = ms_ref["gaussian_predictor.gaussian_queries"].shape
        n_gaussians = q_shape[1]
        d_model = q_shape[2]
    elif "gaussian_predictor.queries" in ms_ref:
        q_shape = ms_ref["gaussian_predictor.queries"].shape
        n_gaussians = q_shape[1]
        d_model = q_shape[2]
    elif "predictor.queries" in ms_ref:
        q_shape = ms_ref["predictor.queries"].shape
        n_gaussians = q_shape[1]
        d_model = q_shape[2]
    
    layers_found = [int(k.split(".")[2]) for k in ms_ref.keys() if k.startswith("encoder.transformer_layers.") or k.startswith("encoder.tf_layers.")]
    if layers_found:
        n_encoder_layers = max(layers_found) + 1

    if "deform_mlp.net.0.weight" in ms_ref:
        d_deform_hidden = ms_ref["deform_mlp.net.0.weight"].shape[0]

    if args.model == "cdg_12":
        # CDG12: n_temporal_bases를 checkpoint의 마지막 Conv 레이어에서 감지
        # basis_conv.0.weight = [d_model//2, d_model, 5] (첫번째 Conv → 이거 아님!)
        # basis_conv.2.weight = [n_bases, d_model//2, 5] (마지막 Conv → 이게 정답!)
        n_temporal_bases = 8
        basis_last_key = "predictor.envelope_gen.basis_conv.2.weight"
        if basis_last_key in ms_ref:
            n_temporal_bases = ms_ref[basis_last_key].shape[0]
        model = ctor(
            d_model=d_model,
            n_gaussians=n_gaussians,
            n_encoder_layers=n_encoder_layers,
            n_temporal_bases=n_temporal_bases,
        ).to(device)
    elif args.model in ("cdg_10", "cdg_11"):
        model = ctor(
            d_model=d_model,
            n_gaussians=n_gaussians,
            n_encoder_layers=n_encoder_layers,
        ).to(device)
    elif args.model in ("cdg_6", "cdg_7", "cdg_8"):
        model = ctor(
            d_model=d_model,
            n_gaussians=n_gaussians,
            n_encoder_layers=n_encoder_layers,
            use_metadata=False,
            direct_lead_sum=args.cdgs_direct_lead_sum,
            direct_lead_patch=args.cdgs_direct_patch,
            direct_lead_span=args.cdgs_direct_span,
            d_deform_hidden=d_deform_hidden,
        ).to(device)
    else:
        model = ctor(
            d_model=d_model,
            n_gaussians=n_gaussians,
            use_metadata=False,
            direct_lead_sum=args.cdgs_direct_lead_sum,
            direct_lead_patch=args.cdgs_direct_patch,
            direct_lead_span=args.cdgs_direct_span,
        ).to(device)
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as e:
        # metadata 키가 ckpt에만 있을 때(우리가 use_metadata=False로 생성했기 때문) → strict=False로 재시도
        msg = str(e)
        if "Unexpected key(s) in state_dict" in msg and ("meta_conditioner" in msg or "meta_electrode_net" in msg):
            model.load_state_dict(ckpt["model_state"], strict=False)
        # legacy cdg_4(128) checkpoint compatibility
        elif args.model == "cdg_4":
            ms = ckpt.get("model_state", {})
            cb = ms.get("pos_codebook", None)
            gq = ms.get("gaussian_predictor.gaussian_queries", None)
            if cb is not None and getattr(cb, "shape", None) == (128, 3) and gq is not None and getattr(gq, "shape", None) == (
                1,
                128,
                256,
            ):
                model = _CDGS4Legacy128().to(device)
                # 레거시 모듈은 encoder/gaussian_predictor만 포함 → 나머지 키는 무시
                model.load_state_dict(ms, strict=False)
            else:
                raise
        else:
            raise
    model.eval()

    batch = None
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i == args.batch_index:
                break
        else:
            raise SystemExit(f"batch_index={args.batch_index} 에 해당하는 배치가 없습니다.")

    x, y, age, sex = batch
    x = x.to(device)
    age = age.to(device)
    sex = sex.to(device)
    T = x.size(2)

    with torch.no_grad():
        if args.model in ("cdg_10", "cdg_11", "cdg_12"):
            meta = torch.stack([age.float(), sex.float(), torch.zeros_like(age), torch.zeros_like(age)], dim=-1).to(device)
            _, _, extras = model(x, meta=meta)
        else:
            _, _, extras = model(x, age, sex)

    if args.model in ("cdg_10", "cdg_11", "cdg_12"):
        # CDGS10/11/12: 실제 3D 위치가 있는 물리 기반 모델!
        amplitude = extras["amplitude"]
        mu = torch.zeros_like(amplitude)
        sigma0 = torch.zeros_like(amplitude)
        sigma_vel = torch.zeros_like(amplitude)
        gaussian_dirs_t = extras.get("dipole_dir_t", None)

        # position: [B, N, 3] → 시간 축으로 확장 (위치는 고정, envelope로 맥박 표현)
        p_hat = extras["position"].unsqueeze(-1).expand(-1, -1, -1, T)
        sigma_t = extras["envelope_t"]
        cov3 = None
        interior = True
    elif args.model == "cdg_8":
        amplitude = extras["amplitude"]
        mu = torch.zeros_like(amplitude) # dummy
        sigma0 = torch.zeros_like(amplitude) # dummy
        sigma_vel = torch.zeros_like(amplitude) # dummy
        gaussian_dirs_t = None
        
        # 모델의 실제 물리적 공간 좌표인 p_pos를 점의 위치(p_hat)로 렌더링합니다!
        p_hat = extras["p_pos"].unsqueeze(-1).expand(-1, -1, -1, T) 
        sigma_t = extras["envelope_t"]
        cov3 = None
        interior = True
    else:
        mu = extras["mu"]
        sigma0 = extras["sigma0"]
        sigma_vel = extras["sigma_vel"]
        amplitude = extras["amplitude"]
        gaussian_dirs_t = None
    cov3 = None
    if "scales" in extras and "rot6d" in extras:
        cov3 = compute_cov3_from_scales_rot6d(extras["scales"], extras["rot6d"])
    if args.model in ("cdg_8", "cdg_10", "cdg_11", "cdg_12"):
        pass # 이미 위에서 p_hat, sigma_t, interior를 모두 세팅함
    elif args.model in ("cdg_4", "cdg_5"):
        # codebook-driven movement
        if "delta_code" in extras and hasattr(model, "delta_mlp"):
            codebook = model.pos_codebook.to(device=device)
            p_hat, sigma_t, _ = compute_p_cellbox_time(
                mu,
                sigma0,
                sigma_vel,
                extras["cell_logits"],
                extras["delta_code"],
                model.delta_mlp,
                codebook,
                extras["pos_temp"],
                bool(getattr(model, "hard_positions", True)),
                T,
            )
        else:
            # legacy delta0/delta_vel
            codebook = getattr(model, "pos_codebook", None)
            if codebook is None:
                codebook = model.pos_codebook
            codebook = codebook.to(device=device)
            p_hat, sigma_t, _ = compute_p_cellbox_time_legacy_delta(
                mu,
                sigma0,
                sigma_vel,
                extras["cell_logits"],
                extras["delta0"],
                extras["delta_vel"],
                codebook,
                extras["pos_temp"],
                hard_positions=True,
                T=T,
            )
        interior = True
    else:
        p0 = extras["p0"]
        # cdg_6: DeformationMLP pre-computes trajectories → use p_pos directly
        if "p_pos" in extras:
            p_hat = extras["p_pos"]  # (B,N,3,T) — already deformed
            t_grid = torch.linspace(0, 1, T, device=device)
            dt = t_grid.view(1, 1, T) - mu.unsqueeze(-1)
            sigma_t = F.softplus(sigma0.unsqueeze(-1) + sigma_vel.unsqueeze(-1) * dt) + 1e-3
        else:
            p_vel = extras.get("p_vel", None)
            interior = args.model in ("cdg_3",)
            p_hat, sigma_t, _ = compute_p_hat_sigma(
                mu,
                sigma0,
                sigma_vel,
                p0,
                p_vel,
                T,
                interior_ball=interior,
            )
        interior = args.model in ("cdg_3", "cdg_6", "cdg_7", "cdg_8", "cdg_10", "cdg_11", "cdg_12")

    if args.model in ("cdg_10", "cdg_11", "cdg_12"):
        # CDGS10/11/12: 9전극 좌표를 리드 표시로 사용
        E = model.renderer.electrode_base.detach().cpu()
        lead_np = E.unsqueeze(-1).expand(-1, -1, T).numpy()  # [9, 3, T]
    else:
        with torch.no_grad():
            lead_vecs_t = compute_lead_vectors_time(model, age, sex, T)
        lead_np = lead_vecs_t.detach().cpu().numpy()[0]

    p_np = p_hat.detach().cpu().numpy()
    s_np = sigma_t.detach().cpu().numpy()
    a_np = amplitude.detach().cpu().numpy()
    m_np = mu.detach().cpu().numpy()
    cov_np = None if cov3 is None else cov3.detach().cpu().numpy()
    dirs_np = None
    if (gaussian_dirs_t is not None) and (not args.no_gaussian_directions):
        dirs_np = gaussian_dirs_t.detach().cpu().numpy()[0]

    sync_top_k = args.top_k if args.top_k > 0 else 0
    sync_m = compute_over_sync_metrics(
        sigma_t=sigma_t,
        amplitude=amplitude,
        top_k=sync_top_k,
        gaussian_dirs_t=gaussian_dirs_t if not args.no_gaussian_directions else None,
    )
    print(
        f"[Sync] slots={sync_m['slots_used']} | "
        f"EnvCos(mean/med/p90)={sync_m['env_cos_mean']:.3f}/{sync_m['env_cos_median']:.3f}/{sync_m['env_cos_p90']:.3f}"
    )
    print(
        f"[Sync] EnvCos>0.8={sync_m['env_frac_gt_08']*100:.1f}% | "
        f"PeakSpread={sync_m['peak_spread_norm']:.3f} | EffRank={sync_m['effective_rank']:.1f}"
    )
    if "dir_resultant_mean" in sync_m:
        print(
            f"[Sync] DirResultant(mean/p90)={sync_m['dir_resultant_mean']:.3f}/{sync_m['dir_resultant_p90']:.3f} | "
            f"DirPairCos(mean)={sync_m['dir_pair_cos_mean']:.3f}"
        )

    if sync_m["env_cos_mean"] > 0.75:
        print("  └─ [Warn] envelope 과동기화 가능성이 큽니다.")
    elif sync_m["env_cos_mean"] > 0.55:
        print("  └─ [Note] envelope 동기화가 다소 높은 편입니다.")

    if "dir_resultant_mean" in sync_m and sync_m["dir_resultant_mean"] > 0.85:
        print("  └─ [Warn] 방향 정렬이 매우 강합니다 (dir 동기화).")

    out_path = Path(args.out_path)
    plot_matplotlib(
        p_np,
        s_np,
        a_np,
        m_np,
        out_path,
        args.top_k,
        args.sphere_alpha,
        lead_vecs_bt=None if args.no_lead_cameras else lead_np,
        interior_ball_trajectory=interior,
        jitter_scale=(0.02 if args.model in ("cdg_4", "cdg_5") else 0.0),
        cov3_bt=cov_np,
        gaussian_dirs_bt=dirs_np,
        direction_scale=args.direction_scale,
        sort_by=args.sort_by,
    )
    print(f"[OK] matplotlib → {out_path.resolve()}")

    if args.html:
        try:
            plot_plotly(
                p_np,
                s_np,
                a_np,
                m_np,
                Path(args.html),
                args.top_k,
                show_unit_sphere=not args.no_plotly_sphere,
                sphere_opacity=args.plotly_sphere_opacity,
                lead_vecs=lead_np,
                show_lead_cameras=not args.no_lead_cameras,
                plotly_time_slider=not args.no_plotly_time_slider,
                max_time_slider_frames=args.plotly_max_time_frames,
                interior_ball_trajectory=interior,
                jitter_scale=(0.02 if args.model in ("cdg_4", "cdg_5") else 0.0),
                cov3_bt=cov_np,
                gaussian_dirs_bt=dirs_np,
                direction_scale=args.direction_scale,
                sort_by=args.sort_by,
            )
            print(f"[OK] plotly → {Path(args.html).resolve()}")
            if not args.no_plotly_time_slider:
                print(
                    "     아래 슬라이더로 시간 t 이동 · ▶ 재생 · 드래그=3D 회전"
                )
            else:
                print(
                    "     보기: 드래그=회전, 스크롤=확대, 범례 클릭=숨김/표시"
                )
        except ImportError as e:
            raise SystemExit("plotly 필요: pip install plotly") from e


if __name__ == "__main__":
    main()
