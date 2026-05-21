# -*- coding: utf-8 -*-
"""학습된 PriorConditionedRecon 모델로 test set 추론.

사용 전제 — 학습 시 저장된 파일들:
  outputs/cascade_prior/
    ├── best.pt          (학습 가중치)
    ├── config.json      (모델 설정)
    ├── test_pairs.csv   (test split의 (past, current, delta_days) 페어)
    └── meta_stats.npz   (use_meta=True일 때만)

저장 결과:
  outputs/cascade_prior/inference/
    ├── reconstructions.npz     # 입력(Lead I) + 과거 12-lead + 복원 + GT
    ├── metrics_per_sample.csv  # 샘플별 lead별 MAE/Pearson
    ├── metrics_summary.csv     # 전체 평균/표준편차
    └── sample_plots/           # (선택) 샘플 시각화

사용 예:
  python infer_cascade_prior.py \
      --data_dir /workspace/physionet.org/files/ptb-xl/1.0.3/ \
      --ckpt_dir ./outputs/cascade_prior \
      --save_plots 10
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cascade_recon import CascadeECGRecon, LEADS, IDX
from train_cascade_prior import (
    PriorConditionedRecon, PTBXLPriorDataset, build_pairs
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/workspace/physionet.org/files/ptb-xl/1.0.3/")
    p.add_argument("--ckpt_dir", default="./outputs/cascade_prior",
                   help="학습 결과 폴더 (best.pt, config.json 등)")
    p.add_argument("--ckpt_name", default="best.pt", help="best.pt 또는 last.pt")
    p.add_argument("--out_dir", default=None,
                   help="추론 결과 저장 폴더 (기본: ckpt_dir/inference)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_plots", type=int, default=10,
                   help="시각화할 샘플 수 (0이면 안 그림)")
    p.add_argument("--test_folds", nargs="+", type=int, default=[10],
                   help="test_pairs.csv 가 없을 때 사용할 strat_fold")
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device, use_meta):
    """test loader 전체에 대해 추론, 결과 수집."""
    model.eval()
    all_inputs, all_targets, all_preds = [], [], []
    all_past, all_td = [], []

    for batch in tqdm(loader, desc="inference"):
        cur_lead_i, cur_target, past_12, td, meta = batch
        cur_lead_i = cur_lead_i.to(device)
        cur_target = cur_target.to(device)
        past_12 = past_12.to(device)
        td = td.to(device)
        meta_t = meta.to(device) if use_meta else None

        pred = model(cur_lead_i, past_12lead=past_12, time_delta=td,
                     patient_meta=meta_t)
        if isinstance(pred, tuple):
            pred = pred[0]

        all_inputs.append(cur_lead_i.cpu().numpy())
        all_targets.append(cur_target.cpu().numpy())
        all_preds.append(pred.cpu().numpy())
        all_past.append(past_12.cpu().numpy())
        all_td.append(td.cpu().numpy())

    return (
        np.concatenate(all_inputs, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_past, axis=0),
        np.concatenate(all_td, axis=0),
    )


def compute_per_sample_metrics(targets, preds):
    """샘플별, lead별 MAE & Pearson 계산.
    targets, preds: (N, 12, L)
    return: DataFrame (N행, 24컬럼: MAE_*, R_*)
    """
    n_samples = targets.shape[0]
    rows = []
    for i in range(n_samples):
        row = {}
        for li in range(12):
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


def plot_sample(input_lead, target_12, pred_12, past_12, info, save_path):
    """원본 vs 복원 vs 과거 비교 그래프."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True)
    for li in range(12):
        ax = axes[li // 2, li % 2]
        ax.plot(target_12[li], label="GT (current)", color="black", linewidth=1.2)
        ax.plot(pred_12[li], label="Pred", color="red", linewidth=1.0, alpha=0.8)
        ax.plot(past_12[li], label="Past", color="green",
                linewidth=0.8, linestyle=":", alpha=0.5)
        if LEADS[li] == "I":
            ax.plot(input_lead[0], label="Input(I)", color="blue",
                    linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(LEADS[li])
        ax.grid(True, alpha=0.3)
        if li == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(info)
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

    # ── test pairs 로드 (없으면 fold 기반으로 빌드) ──
    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    fname_col = "filename_hr" if cfg["use_hr"] else "filename_lr"
    df_meta = df.rename(columns={fname_col: "filename_lr"})

    test_pairs_csv = ckpt_dir / "test_pairs.csv"
    if test_pairs_csv.exists():
        tp = pd.read_csv(test_pairs_csv)
        test_pairs = list(zip(tp["past"].tolist(),
                              tp["current"].tolist(),
                              tp["delta_days"].astype(int).tolist()))
        print(f"[TEST] loaded {len(test_pairs)} pairs from test_pairs.csv")
    else:
        test_pairs, _ = build_pairs(df, fname_col, args.test_folds)
        print(f"[TEST] built {len(test_pairs)} pairs from fold {args.test_folds} "
              f"(test_pairs.csv가 없어서 빌드함)")

    if len(test_pairs) == 0:
        print("[ERROR] test pair가 없습니다.")
        return

    # ── 데이터셋 ──
    if cfg["use_meta"]:
        stats = np.load(ckpt_dir / "meta_stats.npz")
        test_ds = PTBXLPriorDataset(
            data_dir, test_pairs, df_meta=df_meta, use_meta=True,
            h_mean=float(stats["h_mean"]), w_mean=float(stats["w_mean"]),
            h_std=float(stats["h_std"]), w_std=float(stats["w_std"]),
        )
    else:
        test_ds = PTBXLPriorDataset(data_dir, test_pairs, df_meta=df_meta,
                                    use_meta=False)

    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ── 모델 빌드 + 체크포인트 로드 ──
    meta_dim = 6 if cfg["use_meta"] else 0
    total_cond_dim = cfg["prior_dim"] + meta_dim
    base_model = CascadeECGRecon(
        base_ch=cfg["base_ch"],
        res_base_ch=cfg["res_base_ch"],
        residual_scale=cfg["residual_scale"],
        meta_dim=total_cond_dim,
    )
    model = PriorConditionedRecon(
        base_model, prior_dim=cfg["prior_dim"], use_meta=cfg["use_meta"],
    ).to(device)

    ckpt_path = ckpt_dir / args.ckpt_name
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(f"[LOAD] {ckpt_path}")

    # ── 추론 ──
    inputs, targets, preds, past, td = run_inference(
        model, test_loader, device, cfg["use_meta"]
    )
    print(f"[INFER] inputs={inputs.shape}, targets={targets.shape}, preds={preds.shape}")

    # ── 결과 저장 (.npz) ──
    npz_path = out_dir / "reconstructions.npz"
    np.savez_compressed(
        npz_path,
        inputs=inputs,       # (N, 1, L) 현재 Lead I
        targets=targets,     # (N, 12, L) 현재 12-lead GT
        preds=preds,         # (N, 12, L) 복원 12-lead
        past=past,           # (N, 12, L) 과거 12-lead
        time_delta=td,       # (N, 1)    log-정규화 시간차
        pairs=np.array([(p, c, d) for p, c, d in test_pairs]),
        leads=np.array(LEADS),
    )
    print(f"[SAVE] {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")

    # ── 샘플별 metric ──
    df_metrics = compute_per_sample_metrics(targets, preds)
    df_metrics.insert(0, "past_path", [p for p, _, _ in test_pairs])
    df_metrics.insert(1, "current_path", [c for _, c, _ in test_pairs])
    df_metrics.insert(2, "delta_days", [d for _, _, d in test_pairs])
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
    print("  MAE     " + " ".join(f"{L}:{summary[f'MAE_{L}_mean']:.3f}" for L in LEADS))
    print("  Pearson " + " ".join(f"{L}:{summary[f'R_{L}_mean']:.3f}" for L in LEADS))
    print(f"\n  Overall MAE: {summary['MAE_overall']:.4f}")
    print(f"  Overall Pearson: {summary['R_overall']:.4f}")

    # ── 샘플 시각화 ──
    if args.save_plots > 0:
        plot_dir = out_dir / "sample_plots"
        plot_dir.mkdir(exist_ok=True)
        n_plot = min(args.save_plots, len(test_pairs))
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(test_pairs), size=n_plot, replace=False)
        print(f"[PLOT] {n_plot} samples → {plot_dir}")
        for idx in tqdm(idxs, desc="plotting"):
            past_p, cur_p, delta = test_pairs[idx]
            info = f"past={past_p} → cur={cur_p} (Δ{delta}d)"
            safe = f"{idx:05d}"
            plot_sample(inputs[idx], targets[idx], preds[idx], past[idx],
                        info, plot_dir / f"{safe}.png")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
