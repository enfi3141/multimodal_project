# -*- coding: utf-8 -*-
"""
과거 12-lead ECG + 현재 Lead I → 현재 12-lead 복원 (Prior-Conditioned)
- PTB-XL에서 같은 환자의 반복 기록 활용
- 과거 12-lead를 Prior Encoder로 encoding → base 모델에 conditioning
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from new.cascade_recon import (
    CascadeECGRecon, VanillaECGRecon, LEADS, IDX,
    cascade_loss, get_lead_weight_tensor, count_params,
)


# ═══════════════════════════════════════════════════
#  Prior Encoder: 과거 12-lead → conditioning vector
# ═══════════════════════════════════════════════════
class PriorEncoder(nn.Module):
    """과거 12-lead ECG를 압축하여 conditioning vector를 생성.
    out_dim - 1 차원을 ECG에서 추출하고, 1차원은 time_delta용으로 남겨둠."""
    def __init__(self, in_ch=12, out_dim=64):
        super().__init__()
        ecg_dim = out_dim - 1  # 1차원은 time_delta에 할당
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 15, padding=7, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 7, padding=3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, ecg_dim),
        )

    def forward(self, x):
        return self.net(x)


class PriorConditionedRecon(nn.Module):
    """과거 12-lead + 시간 차이 + 환자 메타를 encoding하여 base 모델에 meta로 주입."""
    def __init__(self, base_model, prior_dim=64, use_meta=False):
        super().__init__()
        self.base = base_model
        self.prior_enc = PriorEncoder(in_ch=12, out_dim=prior_dim)
        self.use_meta = use_meta

    def forward(self, lead_i, past_12lead=None, time_delta=None,
                patient_meta=None, gt_full=None, teacher_forcing=0.0,
                return_intermediate=False):
        parts = []

        # 1) 과거 ECG feature + time_delta
        if past_12lead is not None:
            ecg_feat = self.prior_enc(past_12lead)  # (B, prior_dim-1)
            if time_delta is not None:
                parts.append(torch.cat([ecg_feat, time_delta], dim=-1))  # (B, prior_dim)
            else:
                zeros = torch.zeros(ecg_feat.size(0), 1, device=ecg_feat.device)
                parts.append(torch.cat([ecg_feat, zeros], dim=-1))

        # 2) 환자 메타 (나이/성별/키/체중)
        if self.use_meta and patient_meta is not None:
            parts.append(patient_meta)  # (B, 6)

        # concat → base model의 meta로 주입
        if parts:
            conditioning = torch.cat(parts, dim=-1)
        else:
            conditioning = None

        return self.base(
            lead_i,
            gt_full=gt_full,
            teacher_forcing=teacher_forcing,
            meta=conditioning,
            return_intermediate=return_intermediate,
        )


# ═══════════════════════════════════════════════════
#  Dataset: 같은 환자의 (과거 12-lead, 현재 Lead I) 쌍
# ═══════════════════════════════════════════════════
class PTBXLPriorDataset(Dataset):
    """
    같은 환자의 반복 기록에서:
      - past: 이전 방문의 12-lead ECG 전체
      - current: 현재 방문의 Lead I (입력) → 12-lead (타겟)
      - time_delta: 두 기록 간 시간 차이 (일수, log 정규화)
      - meta: 환자 메타 (나이/성별/키/체중) [optional]
    """
    def __init__(self, data_dir, pair_list, df_meta=None, use_meta=False,
                 h_mean=None, w_mean=None, h_std=None, w_std=None):
        """
        pair_list: [(past_path, current_path, delta_days), ...]
        """
        self.items = []
        self.use_meta = use_meta

        if use_meta and df_meta is not None:
            df_idx = df_meta.set_index("filename_lr") if "filename_lr" in df_meta.columns else df_meta
            if h_mean is None:
                valid_h = df_meta["height"].dropna()
                valid_w = df_meta["weight"].dropna()
                self.h_mean = float(valid_h.mean()) if len(valid_h) else 170.0
                self.w_mean = float(valid_w.mean()) if len(valid_w) else 75.0
                self.h_std = float(valid_h.std()) if len(valid_h) > 1 else 10.0
                self.w_std = float(valid_w.std()) if len(valid_w) > 1 else 15.0
            else:
                self.h_mean, self.w_mean = h_mean, w_mean
                self.h_std, self.w_std = h_std, w_std
        else:
            df_idx = None

        for past_p, cur_p, delta_days in tqdm(pair_list, leave=False, desc="loading pairs"):
            # 과거 ECG
            sig_past, _ = wfdb.rdsamp(str(data_dir / past_p))
            sig_past = sig_past.astype(np.float32).T
            m, s = sig_past.mean(), sig_past.std() + 1e-6
            sig_past = (sig_past - m) / s

            # 현재 ECG
            sig_cur, _ = wfdb.rdsamp(str(data_dir / cur_p))
            sig_cur = sig_cur.astype(np.float32).T
            m, s = sig_cur.mean(), sig_cur.std() + 1e-6
            sig_cur = (sig_cur - m) / s

            # 시간 차이: log(1 + days) / log(1 + 3650) 으로 정규화 (0~1)
            time_feat = np.log1p(delta_days) / np.log1p(3650.0)

            # 환자 메타
            meta = np.zeros(6, dtype=np.float32)
            if use_meta and df_idx is not None:
                try:
                    row = df_idx.loc[cur_p]
                    age = float(row.age) if not pd.isna(row.age) else 60.0
                    sex = 1.0 if str(row.sex) in ("1", "F", "1.0") else 0.0
                    h_raw, w_raw = row.height, row.weight
                    h_missing = 1.0 if pd.isna(h_raw) else 0.0
                    w_missing = 1.0 if pd.isna(w_raw) else 0.0
                    height = float(h_raw) if not pd.isna(h_raw) else self.h_mean
                    weight = float(w_raw) if not pd.isna(w_raw) else self.w_mean
                    meta = np.array([
                        age / 100.0, sex,
                        (height - self.h_mean) / self.h_std,
                        (weight - self.w_mean) / self.w_std,
                        h_missing, w_missing,
                    ], dtype=np.float32)
                except Exception:
                    pass

            self.items.append((sig_past, sig_cur, np.float32(time_feat), meta))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        past_full, cur_full, time_feat, meta = self.items[i]
        past_12 = torch.from_numpy(past_full)                    # (12, T)
        cur_lead_i = torch.from_numpy(cur_full[IDX["I"]:IDX["I"]+1])  # (1, T)
        cur_target = torch.from_numpy(cur_full)                   # (12, T)
        time_delta = torch.tensor([time_feat])                    # (1,)
        meta_t = torch.from_numpy(meta)                           # (6,)
        return cur_lead_i, cur_target, past_12, time_delta, meta_t


# ═══════════════════════════════════════════════════
#  Baseline Dataset: 과거 없이 Lead I만 사용 (비교용)
# ═══════════════════════════════════════════════════
class PTBXLBaselineDataset(Dataset):
    """반복 환자의 현재 기록만 사용 (과거 정보 없음) - 공정한 비교용."""
    def __init__(self, data_dir, paths):
        self.signals = []
        for p in tqdm(paths, leave=False, desc="loading baseline"):
            sig, _ = wfdb.rdsamp(str(data_dir / p))
            sig = sig.astype(np.float32).T
            m, s = sig.mean(), sig.std() + 1e-6
            sig = (sig - m) / s
            self.signals.append(sig)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, i):
        full = torch.from_numpy(self.signals[i])
        x = full[IDX["I"]:IDX["I"]+1]
        return x, full, torch.zeros(12, full.shape[-1])  # dummy past


# ═══════════════════════════════════════════════════
#  유틸
# ═══════════════════════════════════════════════════
def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_pairs(df, fname_col, fold_values):
    """
    strat_fold가 fold_values에 해당하는 레코드에서
    같은 환자의 반복 기록을 시간순 페어로 만든다.
    반환: pairs [(past_path, current_path, delta_days), ...], single_paths [path, ...]
    """
    sub = df[df["strat_fold"].isin(fold_values)].copy()
    sub["recording_date"] = pd.to_datetime(sub["recording_date"])
    sub = sub.sort_values(["patient_id", "recording_date"])

    pairs = []
    singles = []
    for pid, grp in sub.groupby("patient_id"):
        paths = grp[fname_col].tolist()
        dates = grp["recording_date"].tolist()
        if len(paths) >= 2:
            for j in range(1, len(paths)):
                delta_days = (dates[j] - dates[j-1]).days
                pairs.append((paths[j-1], paths[j], max(0, delta_days)))
        else:
            singles.append(paths[0])

    return pairs, singles


def per_lead_metrics(model, loader, device, use_prior=True, use_meta=False):
    model.eval()
    mae_sum = torch.zeros(12); pearson_acc = torch.zeros(12); n = 0
    with torch.no_grad():
        for batch in loader:
            x, y, past, td, meta = batch
            x, y = x.to(device), y.to(device)
            past_t = past.to(device) if use_prior else None
            td_t = td.to(device) if use_prior else None
            meta_t = meta.to(device) if use_meta else None
            p = model(x, past_12lead=past_t, time_delta=td_t, patient_meta=meta_t)
            if isinstance(p, tuple):
                p = p[0]
            mae_sum += (p - y).abs().mean(dim=(0, 2)).cpu()
            for li in range(12):
                for b in range(p.size(0)):
                    a, c = p[b, li].cpu().numpy(), y[b, li].cpu().numpy()
                    if a.std() > 1e-8 and c.std() > 1e-8:
                        pearson_acc[li] += np.corrcoef(a, c)[0, 1]
            n += p.size(0)
    return mae_sum / max(1, len(loader)), pearson_acc / max(1, n)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/workspace/physionet.org/files/ptb-xl/1.0.3/")
    p.add_argument("--output_dir", default="./outputs/cascade_prior")
    p.add_argument("--use_hr", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--res_base_ch", type=int, default=16)
    p.add_argument("--residual_scale", type=float, default=0.3)
    p.add_argument("--prior_dim", type=int, default=64,
                   help="과거 12-lead encoding 차원")
    p.add_argument("--teacher_forcing", action="store_true")
    p.add_argument("--tf_decay_ratio", type=float, default=0.5)
    p.add_argument("--use_meta", action="store_true",
                   help="환자 메타 (나이/성별/키/체중)도 conditioning에 추가")
    return p.parse_args()


# ═══════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    fname_col = "filename_hr" if args.use_hr else "filename_lr"

    # ── strat_fold 기반 split (환자 단위, 누수 없음) ──
    train_folds = list(range(1, 9))   # fold 1~8
    val_folds   = [9]
    test_folds  = [10]

    train_pairs, train_singles = build_pairs(df, fname_col, train_folds)
    val_pairs, val_singles     = build_pairs(df, fname_col, val_folds)
    test_pairs, test_singles   = build_pairs(df, fname_col, test_folds)

    print(f"[DATA] train pairs={len(train_pairs)}, singles={len(train_singles)}")
    print(f"[DATA] val   pairs={len(val_pairs)}, singles={len(val_singles)}")
    print(f"[DATA] test  pairs={len(test_pairs)}, singles={len(test_singles)}")

    if len(train_pairs) == 0:
        print("[ERROR] 반복 기록 환자가 train에 없습니다!")
        return

    # 데이터셋
    df_meta = df.rename(columns={fname_col: "filename_lr"})
    train_ds = PTBXLPriorDataset(data_dir, train_pairs, df_meta=df_meta,
                                  use_meta=args.use_meta)
    if val_pairs:
        val_kw = {}
        if args.use_meta:
            val_kw = dict(h_mean=train_ds.h_mean, w_mean=train_ds.w_mean,
                          h_std=train_ds.h_std, w_std=train_ds.w_std)
        val_ds = PTBXLPriorDataset(data_dir, val_pairs, df_meta=df_meta,
                                    use_meta=args.use_meta, **val_kw)
    else:
        val_ds = None

    if args.use_meta:
        np.savez(output_dir / "meta_stats.npz",
                 h_mean=train_ds.h_mean, w_mean=train_ds.w_mean,
                 h_std=train_ds.h_std, w_std=train_ds.w_std)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True) if val_ds else None

    # ── 모델: base CascadeECGRecon + PriorEncoder ──
    # conditioning 차원: prior_dim (ECG+time) + 6 (meta, if use_meta)
    meta_dim = 6 if args.use_meta else 0
    total_cond_dim = args.prior_dim + meta_dim

    base_model = CascadeECGRecon(
        base_ch=args.base_ch,
        res_base_ch=args.res_base_ch,
        residual_scale=args.residual_scale,
        meta_dim=total_cond_dim,
    )
    model = PriorConditionedRecon(base_model, prior_dim=args.prior_dim,
                                   use_meta=args.use_meta).to(device)

    # 설정 저장
    cfg = {
        "base_ch": args.base_ch,
        "res_base_ch": args.res_base_ch,
        "residual_scale": args.residual_scale,
        "prior_dim": args.prior_dim,
        "use_meta": args.use_meta,
        "use_hr": args.use_hr,
        "seed": args.seed,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs) if val_pairs else 0,
        "test_pairs": len(test_pairs) if test_pairs else 0,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    total_params = sum(p.numel() for p in model.parameters())
    prior_params = sum(p.numel() for p in model.prior_enc.parameters())
    print(f"[MODEL] PriorConditioned Cascade | prior_dim={args.prior_dim} "
          f"meta={args.use_meta} total_cond={total_cond_dim}")
    print(f"[PARAMS] total={total_params/1e6:.3f}M | prior_enc={prior_params/1e6:.3f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    lead_w = get_lead_weight_tensor(device=device)

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        if args.teacher_forcing:
            tf_prob = max(0.0, 1.0 - (ep - 1) / max(1, args.epochs * args.tf_decay_ratio))
        else:
            tf_prob = 0.0

        # ── Train ──
        model.train()
        tr_loss = 0.0; n_batch = 0
        for batch in tqdm(train_loader, desc=f"ep{ep} tr", leave=False):
            cur_lead_i, cur_target, past_12, td, meta = batch
            cur_lead_i = cur_lead_i.to(device)
            cur_target = cur_target.to(device)
            past_12 = past_12.to(device)
            td = td.to(device)
            meta_t = meta.to(device) if args.use_meta else None

            with torch.cuda.amp.autocast(enabled=True):
                pred, inter = model(
                    cur_lead_i,
                    past_12lead=past_12,
                    time_delta=td,
                    patient_meta=meta_t,
                    gt_full=cur_target,
                    teacher_forcing=tf_prob,
                    return_intermediate=True,
                )
                loss = cascade_loss(pred, cur_target, intermediate=inter, lead_w=lead_w)

            if torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                tr_loss += loss.item(); n_batch += 1

        # ── Val ──
        vl_loss = 0.0; m = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    cur_lead_i, cur_target, past_12, td, meta = batch
                    cur_lead_i = cur_lead_i.to(device)
                    cur_target = cur_target.to(device)
                    past_12 = past_12.to(device)
                    td = td.to(device)
                    meta_t = meta.to(device) if args.use_meta else None
                    pred, inter = model(
                        cur_lead_i, past_12lead=past_12, time_delta=td,
                        patient_meta=meta_t, return_intermediate=True,
                    )
                    vl_loss += cascade_loss(pred, cur_target,
                                            intermediate=inter, lead_w=lead_w).item()
                    m += 1

            mae_pl, r_pl = per_lead_metrics(model, val_loader, device,
                                            use_prior=True, use_meta=args.use_meta)
            print(f"[E{ep:02d}/{args.epochs}] tr={tr_loss/max(1,n_batch):.4f} "
                  f"vl={vl_loss/max(1,m):.4f} tf={tf_prob:.2f}")
            print("  MAE     " + " ".join(f"{LEADS[i]}:{mae_pl[i]:.3f}" for i in range(12)))
            print("  Pearson " + " ".join(f"{LEADS[i]}:{r_pl[i]:.3f}" for i in range(12)))
        else:
            print(f"[E{ep:02d}/{args.epochs}] tr={tr_loss/max(1,n_batch):.4f} (no val pairs)")

        torch.save(model.state_dict(), output_dir / "last.pt")
        cur = vl_loss / max(1, m) if m > 0 else tr_loss / max(1, n_batch)
        if cur < best_val:
            best_val = cur
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"  └─ [SAVE] best.pt val={best_val:.4f}")

    print("[DONE] Prior-Conditioned training complete.")

    # ── 비교: 과거 정보 있을 때 vs 없을 때 ──
    if val_loader:
        print("\n=== Ablation: Prior vs No-Prior ===")
        model.eval()

        # With prior
        mae_prior, r_prior = per_lead_metrics(model, val_loader, device,
                                               use_prior=True, use_meta=args.use_meta)

        # Without prior (past_12lead = zeros)
        mae_no, r_no = per_lead_metrics(model, val_loader, device,
                                         use_prior=False, use_meta=False)

        print("[With Prior]")
        print("  MAE     " + " ".join(f"{LEADS[i]}:{mae_prior[i]:.3f}" for i in range(12)))
        print("  Pearson " + " ".join(f"{LEADS[i]}:{r_prior[i]:.3f}" for i in range(12)))
        print("[Without Prior]")
        print("  MAE     " + " ".join(f"{LEADS[i]}:{mae_no[i]:.3f}" for i in range(12)))
        print("  Pearson " + " ".join(f"{LEADS[i]}:{r_no[i]:.3f}" for i in range(12)))

        # 차이 요약
        print("\n[Improvement (Prior - NoPrior)]")
        for i in range(12):
            mae_diff = mae_no[i] - mae_prior[i]  # 양수면 prior가 더 좋음
            r_diff = r_prior[i] - r_no[i]         # 양수면 prior가 더 좋음
            print(f"  {LEADS[i]:>4s} | MAE ↓{mae_diff:.4f} | Pearson ↑{r_diff:.4f}")


if __name__ == "__main__":
    main()
