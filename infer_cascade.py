# -*- coding: utf-8 -*-
"""학습된 모델로 test set 추론 + 복원 결과/지표 저장.

학습 시 저장된 파일들:
  - best.pt (또는 last.pt)
  - test_recs.csv
  - config.json
  - meta_stats.npz (use_meta=True일 때)
를 그대로 활용한다.

저장 결과:
  outputs/cascade/inference/
    ├── reconstructions.npz   # 원본 + 복원 + 입력 신호
    ├── metrics_per_sample.csv  # 샘플별 lead별 MAE/Pearson
    ├── metrics_summary.csv     # 전체 평균/표준편차
    └── sample_plots/           # (선택) 일부 샘플 시각화
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader
from tqdm import tqdm

from cascade_recon import CascadeECGRecon, VanillaECGRecon, LEADS, IDX
from train_cascade import PTBXLDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/workspace/physionet.org/files/ptb-xl/1.0.3/")
    p.add_argument("--ckpt_dir", default="./outputs/cascade",
                   help="학습 결과 폴더 (best.pt, test_recs.csv 등)")
    p.add_argument("--ckpt_name", default="best.pt", help="best.pt 또는 last.pt")
    p.add_argument("--out_dir", default=None,
                   help="추론 결과 저장 폴더 (기본: ckpt_dir/inference)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_plots", type=int, default=10,
                   help="시각화할 샘플 수 (0이면 안 그림)")
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device, use_meta):
    """test loader 전체에 대해 추론, 결과 수집."""
    model.eval()

    all_inputs = []   # (N, 1, L)  Lead I
    all_targets = []  # (N, 12, L)
    all_preds = []    # (N, 12, L)
    all_paths = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="inference")):
        x, y, meta = batch
        x, y = x.to(device), y.to(device)
        meta_t = meta.to(device) if use_meta else None
        pred = model(x, meta=meta_t)

        all_inputs.append(x.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        all_preds.append(pred.cpu().numpy())

        # 배치 내 샘플 경로 매칭
        b = x.size(0)
        start = batch_idx * loader.batch_size
        end = start + b
        all_paths.extend(loader.dataset.paths[start:end])

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    return inputs, targets, preds, all_paths


def compute_per_sample_metrics(targets, preds):
    """샘플별, lead별 MAE & Pearson 계산.
    targets, preds: (N, 12, L)
    return: DataFrame (N행, 24컬럼: MAE_*, R_*)
    """
    n_samples, n_leads, _ = targets.shape
    rows = []
    for i in range(n_samples):
        row = {}
        for li in range(n_leads):
            a = preds[i, li]
            b = targets[i, li]
            mae = float(np.abs(a - b).mean())
            if a.std() > 1e-8 and b.std() > 1e-8:
                r = float(np.corrcoef(a, b)[0, 1])
            else:
                r = float("nan")
            row[f"MAE_{LEADS[li]}"] = mae
            row[f"R_{LEADS[li]}"] = r
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sample(input_lead, target_12, pred_12, path, save_path):
    """원본 vs 복원 비교 그래프."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True)
    for li in range(12):
        ax = axes[li // 2, li % 2]
        ax.plot(target_12[li], label="GT", color="black", linewidth=1.2)
        ax.plot(pred_12[li], label="Pred", color="red", linewidth=1.0, alpha=0.8)
        if LEADS[li] == "I":
            ax.plot(input_lead[0], label="Input(I)", color="blue",
                    linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(LEADS[li])
        ax.grid(True, alpha=0.3)
        if li == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Reconstruction: {path}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=80)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── config 로드 ──
    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)
    print(f"[CONFIG] {cfg}")

    # ── test_recs 로드 ──
    test_recs = pd.read_csv(ckpt_dir / "test_recs.csv")["path"].tolist()
    print(f"[TEST] {len(test_recs)} samples")

    # ── 데이터 로드 ──
    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    fname_col = "filename_hr" if cfg["use_hr"] else "filename_lr"
    df_meta = df.rename(columns={fname_col: "filename_lr"})

    if cfg["use_meta"]:
        stats = np.load(ckpt_dir / "meta_stats.npz")
        test_ds = PTBXLDataset(
            data_dir, test_recs, df_meta, use_meta=True,
            h_mean=float(stats["h_mean"]), w_mean=float(stats["w_mean"]),
            h_std=float(stats["h_std"]), w_std=float(stats["w_std"]),
        )
    else:
        test_ds = PTBXLDataset(data_dir, test_recs, df_meta, use_meta=False)

    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ── 모델 빌드 + 체크포인트 로드 ──
    meta_dim = 6 if cfg["use_meta"] else 0
    if cfg["vanilla"]:
        model = VanillaECGRecon(base_ch=cfg["base_ch"], meta_dim=meta_dim).to(device)
    else:
        model = CascadeECGRecon(
            base_ch=cfg["base_ch"], res_base_ch=cfg["res_base_ch"],
            residual_scale=cfg["residual_scale"], meta_dim=meta_dim,
        ).to(device)

    ckpt_path = ckpt_dir / args.ckpt_name
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(f"[LOAD] {ckpt_path}")

    # ── 추론 ──
    inputs, targets, preds, paths = run_inference(model, test_loader, device, cfg["use_meta"])
    print(f"[INFER] inputs={inputs.shape}, targets={targets.shape}, preds={preds.shape}")

    # ── 결과 저장 (.npz) ──
    npz_path = out_dir / "reconstructions.npz"
    np.savez_compressed(
        npz_path,
        inputs=inputs,       # (N, 1, L) Lead I
        targets=targets,     # (N, 12, L) 원본 12-lead
        preds=preds,         # (N, 12, L) 복원 12-lead
        paths=np.array(paths),
        leads=np.array(LEADS),
    )
    print(f"[SAVE] {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")

    # ── 샘플별 metric ──
    df_metrics = compute_per_sample_metrics(targets, preds)
    df_metrics.insert(0, "path", paths)
    metric_csv = out_dir / "metrics_per_sample.csv"
    df_metrics.to_csv(metric_csv, index=False)
    print(f"[SAVE] {metric_csv}")

    # ── 요약 통계 ──
    summary = {}
    for li in range(12):
        L = LEADS[li]
        summary[f"MAE_{L}_mean"] = df_metrics[f"MAE_{L}"].mean()
        summary[f"MAE_{L}_std"] = df_metrics[f"MAE_{L}"].std()
        summary[f"R_{L}_mean"] = df_metrics[f"R_{L}"].mean()
        summary[f"R_{L}_std"] = df_metrics[f"R_{L}"].std()
    summary["MAE_overall"] = df_metrics[[f"MAE_{L}" for L in LEADS]].mean().mean()
    summary["R_overall"] = df_metrics[[f"R_{L}" for L in LEADS]].mean().mean()

    summary_csv = out_dir / "metrics_summary.csv"
    pd.Series(summary).to_csv(summary_csv, header=["value"])
    print(f"[SAVE] {summary_csv}")

    # 콘솔 요약
    print("\n══ Summary (per-lead mean) ══")
    print("  MAE   " + " ".join(f"{L}:{summary[f'MAE_{L}_mean']:.3f}" for L in LEADS))
    print("  Pearson " + " ".join(f"{L}:{summary[f'R_{L}_mean']:.3f}" for L in LEADS))
    print(f"\n  Overall MAE: {summary['MAE_overall']:.4f}")
    print(f"  Overall Pearson: {summary['R_overall']:.4f}")

    # ── 샘플 시각화 ──
    if args.save_plots > 0:
        plot_dir = out_dir / "sample_plots"
        plot_dir.mkdir(exist_ok=True)
        n_plot = min(args.save_plots, len(paths))
        # 무작위 샘플 고정 (재현성)
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(paths), size=n_plot, replace=False)
        print(f"[PLOT] {n_plot} samples → {plot_dir}")
        for idx in tqdm(idxs, desc="plotting"):
            safe_name = paths[idx].replace("/", "_")
            plot_sample(
                inputs[idx], targets[idx], preds[idx],
                paths[idx], plot_dir / f"{safe_name}.png",
            )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
