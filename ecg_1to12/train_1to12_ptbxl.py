#!/usr/bin/env python
"""Train a 1-lead to 12-lead ECG reconstruction model on PTB-XL."""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath('..'))
from my_decoder import Generator1D

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_ptbxl_downloaded(data_dir: Path) -> None:
    marker = data_dir / "ptbxl_database.csv"
    if marker.exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] PTB-XL not found. Downloading to: {data_dir}")
    wfdb.dl_database("ptb-xl", str(data_dir))


class PTBXLReconstructionDataset(Dataset):
    def __init__(self, data_dir: Path, record_paths: List[str], input_idx: int):
        self.data_dir = data_dir
        self.record_paths = record_paths
        self.input_idx = input_idx

    def __len__(self) -> int:
        return len(self.record_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.record_paths[idx]
        signal, _ = wfdb.rdsamp(str(self.data_dir / rec))
        signal = signal.astype(np.float32)

        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True) + 1e-6
        signal = (signal - mean) / std

        signal = signal.T  # [12, T]
        x = signal[self.input_idx : self.input_idx + 1, :]
        y = signal
        return torch.from_numpy(x), torch.from_numpy(y)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 12, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.down1 = nn.Conv1d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)

        self.enc2 = ConvBlock(base_ch * 2, base_ch * 2)
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1)

        self.enc3 = ConvBlock(base_ch * 4, base_ch * 4)
        self.down3 = nn.Conv1d(base_ch * 4, base_ch * 8, kernel_size=4, stride=2, padding=1)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)

        self.up3 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, kernel_size=4, stride=2, padding=1)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out = nn.Conv1d(base_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, input_idx: int) -> torch.Tensor:
    all_lead_l1 = F.l1_loss(pred, target)
    mask = torch.ones(pred.size(1), dtype=torch.bool, device=pred.device)
    mask[input_idx] = False
    non_input_l1 = F.l1_loss(pred[:, mask], target[:, mask])
    return 0.3 * all_lead_l1 + 0.7 * non_input_l1


def mae_excluding_input(pred: torch.Tensor, target: torch.Tensor, input_idx: int) -> float:
    mask = torch.ones(pred.size(1), dtype=torch.bool, device=pred.device)
    mask[input_idx] = False
    return F.l1_loss(pred[:, mask], target[:, mask]).item()


def pearson_excluding_input(pred: torch.Tensor, target: torch.Tensor, input_idx: int) -> float:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    lead_corrs: List[float] = []

    for lead_idx in range(pred_np.shape[1]):
        if lead_idx == input_idx:
            continue
        p = pred_np[:, lead_idx, :].reshape(-1)
        t = target_np[:, lead_idx, :].reshape(-1)
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            continue
        corr = float(np.corrcoef(p, t)[0, 1])
        if not np.isnan(corr):
            lead_corrs.append(corr)

    return float(np.mean(lead_corrs)) if lead_corrs else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    input_idx: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_mae = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(x)
            loss = reconstruction_loss(pred, y, input_idx)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        running_mae += mae_excluding_input(pred, y, input_idx) * x.size(0)

    total = len(loader.dataset)
    return {
        "loss": running_loss / total,
        "mae_no_input": running_mae / total,
    }


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, input_idx: int) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_corr = 0.0

    for x, y in tqdm(loader, desc="valid", leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        loss = reconstruction_loss(pred, y, input_idx)
        running_loss += loss.item() * x.size(0)
        running_mae += mae_excluding_input(pred, y, input_idx) * x.size(0)
        running_corr += pearson_excluding_input(pred, y, input_idx) * x.size(0)

    total = len(loader.dataset)
    return {
        "loss": running_loss / total,
        "mae_no_input": running_mae / total,
        "pearson_no_input": running_corr / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTB-XL 1-lead to 12-lead reconstruction")
    parser.add_argument("--data_dir", type=str, default="./data/ptb-xl")
    parser.add_argument("--output_dir", type=str, default="./outputs/ptbxl_1to12")
    parser.add_argument("--input_lead", type=str, default="I", choices=LEADS)
    parser.add_argument("--use_hr", action="store_true", help="Use 500 Hz signals (filename_hr).")
    parser.add_argument(
        "--val_folds",
        type=str,
        default="10",
        help="Comma-separated PTB-XL strat_fold values used for validation if strat_fold exists.",
    )
    parser.add_argument("--max_records", type=int, default=0, help="0 means use all records.")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        ensure_ptbxl_downloaded(data_dir)

    csv_path = data_dir / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Run without --skip_download or place PTB-XL there manually."
        )

    df = pd.read_csv(csv_path)
    file_col = "filename_hr" if args.use_hr else "filename_lr"

    rng = np.random.default_rng(args.seed)
    if "strat_fold" in df.columns:
        val_folds = {
            int(x.strip())
            for x in args.val_folds.split(",")
            if x.strip()
        }
        val_df = df[df["strat_fold"].isin(val_folds)]
        train_df = df[~df["strat_fold"].isin(val_folds)]

        train_records = train_df[file_col].tolist()
        val_records = val_df[file_col].tolist()
        rng.shuffle(train_records)
        rng.shuffle(val_records)
    else:
        records = df[file_col].tolist()
        rng.shuffle(records)
        train_records, val_records = train_test_split(
            records,
            test_size=args.val_ratio,
            random_state=args.seed,
        )

    if args.max_records > 0:
        train_target = max(1, int(args.max_records * (1.0 - args.val_ratio)))
        val_target = max(1, args.max_records - train_target)
        train_records = train_records[:train_target]
        val_records = val_records[:val_target]

    input_idx = LEADS.index(args.input_lead)
    train_ds = PTBXLReconstructionDataset(data_dir, train_records, input_idx)
    val_ds = PTBXLReconstructionDataset(data_dir, val_records, input_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # 원래 있던 단순한 모델 제거
    # model = UNet1D(in_ch=1, out_ch=12, base_ch=32).to(device)
    
    # 우리가 만든 강력한 모델 연결!!
    model = Generator1D().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    print("=" * 80)
    print(f"Device: {device}")
    print(f"Input lead: {args.input_lead}")
    print(f"Sampling mode: {'HR(500Hz)' if args.use_hr else 'LR(100Hz)'}")
    print(f"Train/Val: {len(train_ds)}/{len(val_ds)}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, input_idx)
        val_metrics = validate(model, val_loader, device, input_idx)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_mae_no_input={train_metrics['mae_no_input']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mae_no_input={val_metrics['mae_no_input']:.4f} "
            f"val_pearson_no_input={val_metrics['pearson_no_input']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, last_path)

        if val_metrics["mae_no_input"] < best_val:
            best_val = val_metrics["mae_no_input"]
            torch.save(checkpoint, best_path)
            print(f"[INFO] Saved best checkpoint: {best_path}")

    print(f"[DONE] Last checkpoint: {last_path}")
    print(f"[DONE] Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
