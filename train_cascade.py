# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cascade_recon import (
    CascadeECGRecon, VanillaECGRecon, LEADS, IDX,
    cascade_loss, get_lead_weight_tensor, count_params,
)


class PTBXLDataset(Dataset):
    """Per-record norm + meta with missing handling."""
    def __init__(self, data_dir, paths, df_meta, use_meta=False,
                 h_mean=None, w_mean=None, h_std=None, w_std=None):
        self.signals = []
        self.metas = []
        self.paths = list(paths)
        self.use_meta = use_meta
        df_idx = df_meta.set_index("filename_lr") if "filename_lr" in df_meta.columns else df_meta

        if use_meta:
            if h_mean is None:
                valid_h = df_meta["height"].dropna()
                valid_w = df_meta["weight"].dropna()
                self.h_mean = float(valid_h.mean()) if len(valid_h) else 170.0
                self.w_mean = float(valid_w.mean()) if len(valid_w) else 75.0
                self.h_std = float(valid_h.std()) if len(valid_h) > 1 else 10.0
                self.w_std = float(valid_w.std()) if len(valid_w) > 1 else 15.0
            else:
                self.h_mean = h_mean
                self.w_mean = w_mean
                self.h_std = h_std
                self.w_std = w_std

        for p in tqdm(paths, leave=False, desc="loading"):
            sig, _ = wfdb.rdsamp(str(data_dir / p))
            sig = sig.astype(np.float32).T
            mean = sig.mean(); std = sig.std() + 1e-6
            sig = (sig - mean) / std
            self.signals.append(sig)

            if use_meta:
                try:
                    row = df_idx.loc[p]
                    age = float(row.age) if not pd.isna(row.age) else 60.0
                    sex = 1.0 if str(row.sex) in ("1", "F", "1.0") else 0.0
                    h_raw, w_raw = row.height, row.weight
                    h_missing = 1.0 if pd.isna(h_raw) else 0.0
                    w_missing = 1.0 if pd.isna(w_raw) else 0.0
                    height = float(h_raw) if not pd.isna(h_raw) else self.h_mean
                    weight = float(w_raw) if not pd.isna(w_raw) else self.w_mean
                except Exception:
                    age, sex = 60.0, 0.0
                    height, weight = self.h_mean, self.w_mean
                    h_missing, w_missing = 1.0, 1.0

                meta = np.array([
                    age / 100.0,
                    sex,
                    (height - self.h_mean) / self.h_std,
                    (weight - self.w_mean) / self.w_std,
                    h_missing,
                    w_missing,
                ], dtype=np.float32)
                self.metas.append(meta)

    def __len__(self): return len(self.signals)

    def __getitem__(self, i):
        full = torch.from_numpy(self.signals[i])
        x = full[IDX["I"]:IDX["I"]+1]
        y = full
        if self.use_meta:
            return x, y, torch.from_numpy(self.metas[i])
        return x, y, torch.zeros(0)


def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/workspace/physionet.org/files/ptb-xl/1.0.3/")
    p.add_argument("--output_dir", default="./outputs/cascade")
    p.add_argument("--use_hr", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vanilla", action="store_true", help="Vanilla UNet baseline")
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--res_base_ch", type=int, default=16)
    p.add_argument("--residual_scale", type=float, default=0.3)
    p.add_argument("--teacher_forcing", action="store_true")
    p.add_argument("--tf_decay_ratio", type=float, default=0.5)
    p.add_argument("--use_meta", action="store_true")
    return p.parse_args()


def per_lead_metrics(model, loader, device, use_meta):
    model.eval()
    mae_sum = torch.zeros(12); pearson_acc = torch.zeros(12); n = 0
    with torch.no_grad():
        for batch in loader:
            x, y, meta = batch
            x, y = x.to(device), y.to(device)
            meta_t = meta.to(device) if use_meta else None
            p = model(x, meta=meta_t)
            mae_sum += (p - y).abs().mean(dim=(0, 2)).cpu()
            for li in range(12):
                for b in range(p.size(0)):
                    a, c = p[b, li].cpu().numpy(), y[b, li].cpu().numpy()
                    if a.std() > 1e-8 and c.std() > 1e-8:
                        pearson_acc[li] += np.corrcoef(a, c)[0, 1]
            n += p.size(0)
    return mae_sum / max(1, len(loader)), pearson_acc / max(1, n)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    fname_col = "filename_hr" if args.use_hr else "filename_lr"
    recs = df[fname_col].tolist()

    # ── 8 : 1 : 1 split ──
    train_recs, temp_recs = train_test_split(recs, test_size=0.2, random_state=args.seed)
    val_recs, test_recs = train_test_split(temp_recs, test_size=0.5, random_state=args.seed)

    if args.subsample < 1.0:
        train_recs = train_recs[:max(1, int(len(train_recs) * args.subsample))]
        val_recs = val_recs[:max(1, int(len(val_recs) * args.subsample))]
        test_recs = test_recs[:max(1, int(len(test_recs) * args.subsample))]

    print(f"[SPLIT] train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}")

    # 추론에서 재사용할 split 정보 저장
    pd.DataFrame({"path": train_recs}).to_csv(output_dir / "train_recs.csv", index=False)
    pd.DataFrame({"path": val_recs}).to_csv(output_dir / "val_recs.csv", index=False)
    pd.DataFrame({"path": test_recs}).to_csv(output_dir / "test_recs.csv", index=False)
    print(f"[SAVE] splits → {output_dir}")

    df_meta = df.rename(columns={fname_col: "filename_lr"})

    train_ds = PTBXLDataset(data_dir, train_recs, df_meta, use_meta=args.use_meta)

    if args.use_meta:
        val_ds = PTBXLDataset(data_dir, val_recs, df_meta, use_meta=True,
                              h_mean=train_ds.h_mean, w_mean=train_ds.w_mean,
                              h_std=train_ds.h_std, w_std=train_ds.w_std)
        # 메타 통계 저장 (추론 시 재사용)
        np.savez(output_dir / "meta_stats.npz",
                 h_mean=train_ds.h_mean, w_mean=train_ds.w_mean,
                 h_std=train_ds.h_std, w_std=train_ds.w_std)
    else:
        val_ds = PTBXLDataset(data_dir, val_recs, df_meta, use_meta=False)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    meta_dim = 6 if args.use_meta else 0

    if args.vanilla:
        model = VanillaECGRecon(base_ch=args.base_ch, meta_dim=meta_dim).to(device)
    else:
        model = CascadeECGRecon(base_ch=args.base_ch, res_base_ch=args.res_base_ch,
                                 residual_scale=args.residual_scale,
                                 meta_dim=meta_dim).to(device)

    # 학습 설정 저장 (추론 시 재사용)
    cfg = {
        "vanilla": args.vanilla,
        "base_ch": args.base_ch,
        "res_base_ch": args.res_base_ch,
        "residual_scale": args.residual_scale,
        "use_meta": args.use_meta,
        "use_hr": args.use_hr,
        "seed": args.seed,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[MODEL] {'Vanilla' if args.vanilla else 'Cascade'} "
          f"base_ch={args.base_ch} res_base_ch={args.res_base_ch} "
          f"meta={args.use_meta} TF={args.teacher_forcing}")
    print(f"[PARAMS] {count_params(model)/1e6:.3f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    lead_w = get_lead_weight_tensor(device=device)

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        if args.teacher_forcing and not args.vanilla:
            tf_prob = max(0.0, 1.0 - (ep - 1) / max(1, args.epochs * args.tf_decay_ratio))
        else:
            tf_prob = 0.0

        model.train()
        tr_loss = 0.0; n_batch = 0
        for batch in tqdm(train_loader, desc=f"ep{ep} tr (tf={tf_prob:.2f})", leave=False):
            x, y, meta = batch
            x, y = x.to(device), y.to(device)
            meta_t = meta.to(device) if args.use_meta else None
            with torch.cuda.amp.autocast(enabled=True):
                pred, inter = model(x, gt_full=y, teacher_forcing=tf_prob,
                                     meta=meta_t, return_intermediate=True)
                loss = cascade_loss(pred, y, intermediate=inter, lead_w=lead_w)
            if torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                tr_loss += loss.item(); n_batch += 1

        model.eval()
        vl_loss = 0.0; m = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, meta = batch
                x, y = x.to(device), y.to(device)
                meta_t = meta.to(device) if args.use_meta else None
                pred, inter = model(x, meta=meta_t, return_intermediate=True)
                vl_loss += cascade_loss(pred, y, intermediate=inter, lead_w=lead_w).item()
                m += 1

        mae_pl, r_pl = per_lead_metrics(model, val_loader, device, args.use_meta)
        print(f"[E{ep:02d}/{args.epochs}] tr={tr_loss/max(1,n_batch):.4f} vl={vl_loss/max(1,m):.4f} tf={tf_prob:.2f}")
        print("  MAE   " + " ".join(f"{LEADS[i]}:{mae_pl[i]:.3f}" for i in range(12)))
        print("  Pearson " + " ".join(f"{LEADS[i]}:{r_pl[i]:.3f}" for i in range(12)))

        torch.save(model.state_dict(), output_dir / "last.pt")
        cur = vl_loss / max(1, m)
        if cur < best_val:
            best_val = cur
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"  └─ [SAVE] best.pt vl={best_val:.4f}")
    print("[DONE]")


if __name__ == "__main__":
    main()