#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CDGS (또는 동일 입력 형식) 체크포인트로 검증 샘플 몇 개를 골라
정답 12리드 vs 예측 12리드를 한 figure에 겹쳐 그립니다.
각 샘플마다 렌더러와 동일한 시간축 가우시안 인벨로프 히트맵을
sample_XX_*_gaussians.png 로 추가 저장 (--no_gaussian_viz 로 끔).

로컬 (ecg_1to12 안에서):
  python3 visualize_cdgs_predictions.py --checkpoint ../outputs/cdgs/best.pt \\
      --data_dir ./data/ptb-xl --out_dir ./viz_out --num_figures 3 --model cdgs

Docker — 레포 루트에서 셸 열고, **아래 한 줄을 통째로 복사**해서 실행 (백슬래시로 줄 나누지 않아도 됨).
  바꿀 것: device=6 → GPU 번호, PTBXL 호스트 경로, outputs/.../best.pt, --model (cdgs|cdg_2|cdg_3).
  학습 때 --use_hr / --cdgs_direct_* 썼으면 명령 끝에 그대로 붙이기.

  docker run --rm --gpus '"device=6"' -v ~/project_jyu/physionet.org/files/ptb-xl/1.0.3:/workspace/data/ptb-xl -v "$(pwd)":/workspace -w /workspace pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime bash -c "pip install -q wfdb pandas scipy scikit-learn tqdm matplotlib 'numpy<2' && python ecg_1to12/visualize_cdgs_predictions.py --checkpoint outputs/cdg_3/best.pt --data_dir ./data/ptb-xl --out_dir ./viz_out --num_figures 3 --model cdg_3"

의존성: torch, wfdb, pandas, numpy, matplotlib, scikit-learn (학습과 동일)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

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
    if args.max_val > 0:
        val_records = val_records[: args.max_val]
    return val_records


def plot_one_sample(
    target: np.ndarray,
    pred: np.ndarray,
    input_idx: int,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    n_lead, T = target.shape
    assert n_lead == 12 and pred.shape == target.shape
    t = np.arange(T)
    fig, axes = plt.subplots(12, 1, figsize=(12, 22), sharex=True)
    fig.suptitle(title, fontsize=11)
    for k in range(12):
        ax = axes[k]
        ax.plot(t, target[k], label="GT (z-score)", color="C0", linewidth=0.9, alpha=0.85)
        ax.plot(t, pred[k], label="Pred", color="C1", linewidth=0.9, alpha=0.85)
        tag = f"{_LEADS[k]}"
        if k == input_idx:
            tag += "  [INPUT]"
        ax.set_ylabel(tag, rotation=0, labelpad=28, fontsize=9, va="center")
        ax.grid(True, alpha=0.25)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time sample")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _temporal_gaussian_envelope(
    mu: torch.Tensor,
    sigma0: torch.Tensor,
    sigma_vel: torch.Tensor,
    amplitude: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """GaussianRenderer와 동일한 1D 시간축 인벨로프 (N, T). mu·t는 [0,1] 정규화."""
    device, dtype = mu.device, mu.dtype
    t = torch.linspace(0, 1, T, device=device, dtype=dtype).view(1, T)
    mu_bc = mu.unsqueeze(-1)
    dt = t - mu_bc
    sigma_t = F.softplus(sigma0.unsqueeze(-1) + sigma_vel.unsqueeze(-1) * dt) + 1e-3
    return amplitude.unsqueeze(-1) * torch.exp(-0.5 * (dt / sigma_t) ** 2)


def plot_gaussian_extras_sample(
    mu: torch.Tensor,
    sigma0: torch.Tensor,
    sigma_vel: torch.Tensor,
    amplitude: torch.Tensor,
    T: int,
    out_path: Path,
    title: str,
    top_k_curves: int = 8,
) -> None:
    import matplotlib.pyplot as plt

    mu = mu.detach().float().cpu()
    sigma0 = sigma0.detach().float().cpu()
    sigma_vel = sigma_vel.detach().float().cpu()
    amplitude = amplitude.detach().float().cpu()
    gauss_nt = _temporal_gaussian_envelope(mu, sigma0, sigma_vel, amplitude, T).numpy()
    N, _ = gauss_nt.shape
    t_idx = np.arange(T)
    t_norm = np.linspace(0.0, 1.0, T)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1.15, 1.0]})
    ax0, ax1 = axes

    im = ax0.imshow(
        gauss_nt,
        aspect="auto",
        origin="lower",
        extent=(-0.5, T - 0.5, -0.5, N - 0.5),
        cmap="magma",
        interpolation="nearest",
    )
    ax0.set_ylabel("gaussian index n")
    ax0.set_xlabel("time sample")
    ax0.set_title(f"{title}\n(temporal envelope per Gaussian, same as renderer g_n(t))")
    fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.02, label="g_n(t)")

    mu_np = mu.numpy()
    amp_np = amplitude.numpy()
    ax0.scatter(
        mu_np * (T - 1),
        np.arange(N),
        c="cyan",
        s=np.clip(amp_np * 80.0, 8.0, 120.0),
        alpha=0.65,
        edgecolors="white",
        linewidths=0.4,
        label=r"$\mu_n$ (center)",
        zorder=5,
    )
    ax0.legend(loc="upper right", fontsize=8)

    summed = gauss_nt.sum(axis=0)
    ax1.fill_between(t_idx, 0.0, summed, alpha=0.25, color="C0", label=r"$\sum_n g_n(t)$")
    ax1.plot(t_idx, summed, color="C0", linewidth=1.2)

    peak_order = np.argsort(-gauss_nt.max(axis=1))[: min(top_k_curves, N)]
    for j, n in enumerate(peak_order):
        ax1.plot(
            t_idx,
            gauss_nt[n],
            alpha=0.55,
            linewidth=0.9,
            label=f"n={int(n)}" if j < 6 else None,
        )

    ax1.set_xlabel("time sample")
    ax1.set_ylabel("envelope")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=7, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="best.pt 또는 last.pt")
    parser.add_argument(
        "--model",
        type=str,
        default="cdgs",
        choices=("cdgs", "cdg_2", "cdg_3"),
        help="체크포인트와 동일한 아키텍처 (cdgs / cdg_2=CDGS2 / cdg_3=CDGS3)",
    )
    parser.add_argument("--data_dir", type=str, default="./data/ptb-xl")
    parser.add_argument("--out_dir", type=str, default="./viz_cdgs")
    parser.add_argument("--input_lead", type=str, default="I", choices=_LEADS)
    parser.add_argument("--use_hr", action="store_true")
    parser.add_argument("--val_folds", type=str, default="10")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_val", type=int, default=500, help="검증 레코드 최대 개수(빠른 순회)")
    parser.add_argument("--num_figures", type=int, default=4, help="저장할 샘플 수")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cdgs_direct_lead_sum", action="store_true")
    parser.add_argument("--cdgs_direct_patch", type=int, default=5)
    parser.add_argument("--cdgs_direct_span", type=float, default=0.55)
    parser.add_argument(
        "--no_gaussian_viz",
        action="store_true",
        help="시간축 가우시안 히트맵/곡선 PNG 저장 생략",
    )
    parser.add_argument(
        "--gaussian_top_k",
        type=int,
        default=8,
        help="하단 서브플롯에 겹쳐 그릴 상위 K개 가우시안 곡선",
    )
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise SystemExit("matplotlib 필요: pip install matplotlib") from e

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ecg_models"))
    from models import CDGS, CDGS2, CDGS3  # noqa: E402

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    input_idx = _LEADS.index(args.input_lead)
    val_records = build_val_records(args)
    val_ds = PTBXLReconstructionDataset(data_dir, val_records, input_idx)
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    ctor = CDGS if args.model == "cdgs" else CDGS2 if args.model == "cdg_2" else CDGS3
    model = ctor(
        d_model=256,
        n_gaussians=128 if args.model in ("cdg_2", "cdg_3") else 64,
        use_metadata=True,
        direct_lead_sum=args.cdgs_direct_lead_sum,
        direct_lead_patch=args.cdgs_direct_patch,
        direct_lead_span=args.cdgs_direct_span,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    out_dir = Path(args.out_dir)
    saved = 0
    rec_offset = 0
    with torch.no_grad():
        for x, y, age, sex in loader:
            if saved >= args.num_figures:
                break
            x = x.to(device)
            y = y.to(device)
            age = age.to(device)
            sex = sex.to(device)
            pred_fine, _, _ = model(x, age, sex)
            pred_np = pred_fine.cpu().numpy()
            tgt_np = y.cpu().numpy()
            B = x.size(0)
            for i in range(B):
                if saved >= args.num_figures:
                    break
                idx = rec_offset + i
                rec_name = val_records[idx] if idx < len(val_records) else str(idx)
                safe = str(rec_name).replace("/", "_").replace("\\", "_")[:80]
                title = f"{safe} | fold val (train과 동일 split) | input={args.input_lead}"
                plot_one_sample(
                    tgt_np[i],
                    pred_np[i],
                    input_idx,
                    out_dir / f"sample_{saved:02d}_{safe}.png",
                    title=title,
                )
                saved += 1
            rec_offset += B

    extra = "" if args.no_gaussian_viz else f" (+ {saved} gaussian panels)"
    print(f"[DONE] {saved} waveform figures{extra} → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
