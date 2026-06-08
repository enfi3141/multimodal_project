#!/usr/bin/env python3
"""
ResNet1D_Wang — 1D ECG classifier (PTB-XL benchmark).

Faithful re-implementation of resnet1d_wang from:
  Strodthoff et al. (2021) "Deep Learning for ECG Analysis:
  Benchmarks and Insights from PTB-XL"
  github.com/helme/ecg_ptbxl_benchmarking

Architecture:
  - 3-stage ResNet (BasicBlock × [1, 1, 1])
  - 128 channels throughout, no stem stride, no max-pool
  - Kernel sizes: stem=7, block=[5, 3]
  - Head: AdaptiveConcatPool → BN/Dropout/Linear

Usage:
    import torch
    from resnet1d_wang import ResNet1dWang

    model = ResNet1dWang(n_leads=12, n_classes=5)
    x = torch.randn(8, 12, 1000)   # (batch, leads, time)
    logits = model(x)               # (8, 5)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Pooling head ──────────────────────────────────────────────────────────────

class AdaptiveConcatPool1d(nn.Module):
    """Adaptive avg-pool + adaptive max-pool concatenated along channel dim."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            F.adaptive_avg_pool1d(x, 1),
            F.adaptive_max_pool1d(x, 1),
        ], dim=1).squeeze(-1)  # -> (B, 2*C)


def _bn_drop_lin(n_in: int, n_out: int, bn: bool, p: float,
                 actn: Optional[nn.Module]) -> List[nn.Module]:
    layers: List[nn.Module] = []
    if bn: layers.append(nn.BatchNorm1d(n_in))
    if p:  layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


def _create_head1d(nf: int, n_classes: int,
                   lin_ftrs: Optional[List[int]] = None,
                   ps: float = 0.5) -> nn.Sequential:
    nf_in = 2 * nf
    lin_ftrs = [nf_in, n_classes] if lin_ftrs is None \
               else [nf_in] + lin_ftrs + [n_classes]
    ps_list = [ps / 2] * (len(lin_ftrs) - 2) + [ps]
    actns   = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    layers: List[nn.Module] = [AdaptiveConcatPool1d()]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps_list, actns):
        layers += _bn_drop_lin(ni, no, bn=True, p=p, actn=actn)
    return nn.Sequential(*layers)


# ── Backbone ──────────────────────────────────────────────────────────────────

def _conv1d(ni: int, no: int, ks: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(ni, no, ks, stride=stride, padding=ks // 2, bias=False)


class _BasicBlock1dWang(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 kernel_size: List[int] = None, downsample=None):
        super().__init__()
        if kernel_size is None: kernel_size = [5, 3]
        self.conv1      = _conv1d(inplanes, planes, kernel_size[0], stride)
        self.bn1        = nn.BatchNorm1d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = _conv1d(planes, planes, kernel_size[1])
        self.bn2        = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet1dWang(nn.Module):
    """resnet1d_wang: 3-stage 1D ResNet with 128 channels, no stride in stem.

    Args:
        n_leads:          number of input leads (default: 12)
        n_classes:        number of output classes (default: 5)
        inplanes:         base channel width (default: 128)
        kernel_size:      block kernel sizes [k1, k2] (default: [5, 3])
        kernel_size_stem: stem kernel size (default: 7)
    """

    def __init__(self, n_leads: int = 12, n_classes: int = 5,
                 inplanes: int = 128, kernel_size: List[int] = None,
                 kernel_size_stem: int = 7):
        super().__init__()
        if kernel_size is None: kernel_size = [5, 3]
        self.stem = nn.Sequential(
            _conv1d(n_leads, inplanes, kernel_size_stem, stride=1),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
        )
        self._inplanes = inplanes
        self.layer1 = self._make_layer(inplanes, 1, stride=1, ks=kernel_size)
        self.layer2 = self._make_layer(inplanes, 1, stride=2, ks=kernel_size)
        self.layer3 = self._make_layer(inplanes, 1, stride=2, ks=kernel_size)
        self.head   = _create_head1d(inplanes, n_classes, lin_ftrs=[128], ps=0.5)

    def _make_layer(self, planes: int, n_blocks: int,
                    stride: int, ks: List[int]) -> nn.Sequential:
        downsample = None
        if stride != 1 or self._inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self._inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = [_BasicBlock1dWang(self._inplanes, planes, stride, ks, downsample)]
        self._inplanes = planes
        for _ in range(1, n_blocks):
            layers.append(_BasicBlock1dWang(self._inplanes, planes, 1, ks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_leads, T)  -- float32, z-score normalized per lead
        Returns:
            logits: (B, n_classes)  -- apply sigmoid for probabilities
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


# ── Dataset split ─────────────────────────────────────────────────────────────
#
# PTB-XL 10-fold stratified split (Strodthoff et al. 2021 standard protocol):
#
#   Train : folds 1-8  (exclude_multi_recording=True)
#   Val   : fold  9    (exclude_multi_recording=True)
#   Test  : fold  10   (patients with >= 2 recordings -> consecutive pairs)
#
# exclude_multi_recording=True:
#   Only patients who appear exactly once in the entire PTB-XL database are
#   used for classifier training/validation. This prevents any patient seen
#   by the reconstruction model (as a past/current pair) from leaking into
#   the classifier training set.
#
# Paired test set (fold 10):
#   Among fold-10 patients with >= 2 recordings, consecutive pairs
#   (past recording, current recording) form each test sample.
#
# Label set: NORM, MI, STTC, CD, HYP  (PTB-XL superdiagnostic classes)
#   Assigned via scp_statements.csv diagnostic_class field.
#   Samples with no matched label are excluded.
#
# Input:
#   - filename_hr (500 Hz), crop to first 1000 samples (2 seconds)
#   - Per-lead z-score normalization:  x = (x - mean) / (std + 1e-8)
#
# Training:
#   - Optimizer : AdamW  (lr=1e-3, weight_decay=1e-4)
#   - Scheduler : OneCycleLR  (max_lr=1e-3)
#   - Loss      : BCEWithLogitsLoss  (multilabel)
#   - Epochs    : 60,  batch size: 64,  grad clip: 1.0
#   - Best checkpoint selected by macro AUROC on fold-9 val set

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]


def _load_ecg(path: str, crop_len: int = 1000):
    import wfdb, numpy as np
    sig = wfdb.rdrecord(path).p_signal.astype(np.float32).T  # (12, T)
    sig = sig[:, :crop_len]
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-8)
    return sig


def build_test_pairs(data_dir, labels: list, crop_len: int = 1000):
    """
    Builds the exact fold-10 paired test set used in the experiments.
    Fold-10 patients with >= 2 recordings: each consecutive pair is one sample.
    Only samples with at least one diagnostic label are included.
    """
    import ast, numpy as np, pandas as pd
    from pathlib import Path
    data_dir = Path(data_dir)

    scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    scp_map = {str(code): str(row.get("diagnostic_class", ""))
               for code, row in scp.iterrows()
               if bool(row.get("diagnostic", False))}

    db = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    fold10 = db[db["strat_fold"] == 10].copy()

    if "recording_date" in fold10.columns:
        fold10["recording_date"] = pd.to_datetime(fold10["recording_date"], errors="coerce")
        fold10 = fold10.sort_values(["patient_id", "recording_date", "ecg_id"])
    else:
        fold10 = fold10.sort_values(["patient_id", "ecg_id"])

    samples = []
    for _, grp in fold10.groupby("patient_id"):
        grp = grp.reset_index(drop=True)
        if len(grp) < 2:
            continue
        for j in range(1, len(grp)):
            row = grp.iloc[j]
            try:
                codes = ast.literal_eval(row["scp_codes"]) if isinstance(row["scp_codes"], str) else row["scp_codes"]
            except Exception:
                codes = {}
            y = np.zeros(len(labels), dtype=np.float32)
            has_label = False
            for code, score in codes.items():
                if float(score) <= 0:
                    continue
                cls = str(code) if str(code) in labels else scp_map.get(str(code), "")
                if cls in labels:
                    y[labels.index(cls)] = 1.0
                    has_label = True
            if not has_label:
                continue
            # remove extension if present (.hea)
            ecg_path = str(data_dir / row["filename_hr"]).replace(".hea", "")
            samples.append((ecg_path, y))

    print(f"[test set] fold-10 paired samples with labels: {len(samples)}")
    return samples


if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    from sklearn.metrics import roc_auc_score

    # =========================================================================
    # Set paths here
    DATA_DIR  = Path("/path/to/ptb-xl")   # PTB-XL root (contains ptbxl_database.csv)
    CKPT_PATH = "resnet1d_wang_best.pt"   # path to checkpoint file
    # =========================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model
    model = ResNet1dWang(n_leads=12, n_classes=5)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.to(device).eval()

    # 2. Build fold-10 test set (exact same split used in experiments)
    samples = build_test_pairs(DATA_DIR, LABELS, crop_len=1000)

    # 3. Run inference on all test samples
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for ecg_path, y in samples:
            try:
                ecg = _load_ecg(ecg_path, crop_len=1000)
            except Exception as e:
                print(f"[skip] {ecg_path}: {e}")
                continue
            x = torch.tensor(ecg).unsqueeze(0).to(device)  # (1, 12, 1000)
            prob = torch.sigmoid(model(x)).cpu().numpy()    # (1, 5)
            y_true_all.append(y)
            y_prob_all.append(prob[0])

    y_true = np.stack(y_true_all)
    y_prob = np.stack(y_prob_all)

    # 4. Report AUROC per class and macro
    print(f"\n{'Label':<8} {'AUROC':>7}")
    print("-" * 17)
    aucs = []
    for i, label in enumerate(LABELS):
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        aucs.append(auc)
        print(f"{label:<8} {auc:.4f}")
    print(f"{'Macro':<8} {np.mean(aucs):.4f}")
