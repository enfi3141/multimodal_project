#!/usr/bin/env python3
"""
ResNet1dWangGated — ResNet1dWang backbone with gated meta fusion.

Diagram (from paper figure):
    h_recon (128-d)  ──┐
                        ├─ Concat(256-d) → MLP → Sigmoid → g (128-d)
    h_meta  (128-d)  ──┘
    h_fused = g ⊙ h_recon + (1-g) ⊙ h_meta  →  head_cls  →  logits

Usage:
    model = ResNet1dWangGated(n_leads=12, n_classes=5, meta_in_dim=3)
    x    = torch.randn(8, 12, 1000)   # (B, leads, T)
    meta = torch.randn(8, 3)          # (B, meta_dim)  e.g. [age_norm, sex_m, sex_f]
    logits = model(x, meta)           # (8, 5)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet1d_wang import ResNet1dWang


class ResNet1dWangGated(nn.Module):
    """ResNet1dWang backbone with gated ECG-meta feature fusion.

    The ResNet1dWang head is split at the 128-d feature point:

        head_feat : AdaptiveConcatPool → BN → Dropout → Linear(256→128) → ReLU
        head_cls  : BN → Dropout → Linear(128→n_classes)

    Gate fusion is inserted between head_feat and head_cls:

        h_recon = head_feat(backbone(x))          # (B, 128)
        h_meta  = meta_encoder(meta)              # (B, 128)
        g       = sigmoid(MLP([h_recon; h_meta])) # (B, 128)
        h_fused = g * h_recon + (1-g) * h_meta   # (B, 128)
        logits  = head_cls(h_fused)               # (B, n_classes)

    Args:
        n_leads:       number of ECG input leads (default: 12)
        n_classes:     number of output classes  (default: 5)
        meta_in_dim:   dimension of meta input   (default: 3 = age_norm, sex_m, sex_f)
        feat_dim:      ECG feature dimension — must match ResNet1dWang inplanes (default: 128)
        inplanes:      ResNet1dWang base channel width (default: 128)
        kernel_size:   block kernel sizes [k1, k2] (default: [5, 3])
        kernel_size_stem: stem kernel size (default: 7)
    """

    def __init__(
        self,
        n_leads: int = 12,
        n_classes: int = 5,
        meta_in_dim: int = 3,
        feat_dim: int = 128,
        inplanes: int = 128,
        kernel_size: Optional[List[int]] = None,
        kernel_size_stem: int = 7,
    ):
        super().__init__()

        self.feat_dim = feat_dim

        # ── ECG backbone (from ResNet1dWang) ──────────────────────────────────
        base = ResNet1dWang(
            n_leads=n_leads,
            n_classes=n_classes,
            inplanes=inplanes,
            kernel_size=kernel_size,
            kernel_size_stem=kernel_size_stem,
        )

        self.stem   = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        # head children order:
        #   [0] AdaptiveConcatPool1d
        #   [1] BN(256)  [2] Dropout(0.25)  [3] Linear(256→128)  [4] ReLU
        #   [5] BN(128)  [6] Dropout(0.50)  [7] Linear(128→n_classes)
        head_layers = list(base.head.children())
        self.head_feat = nn.Sequential(*head_layers[:5])   # → (B, feat_dim=128)
        self.head_cls  = nn.Sequential(*head_layers[5:])   # → (B, n_classes)

        # ── Meta encoder ──────────────────────────────────────────────────────
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feat_dim),
        )

        # ── Gate ──────────────────────────────────────────────────────────────
        # input: concat(h_recon, h_meta) → 256-d
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )

    # ── Feature extraction ────────────────────────────────────────────────────

    def extract_ecg_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-d ECG features before gate fusion.

        Args:
            x: (B, n_leads, T)
        Returns:
            h_recon: (B, feat_dim)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head_feat(x)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, n_leads, T)   — z-score normalized per lead
            meta : (B, meta_in_dim)  — e.g. [age_norm, sex_male, sex_female]
        Returns:
            logits: (B, n_classes)   — apply sigmoid for probabilities
        """
        h_recon = self.extract_ecg_feat(x)                      # (B, 128)
        h_meta  = self.meta_encoder(meta)                        # (B, 128)

        g       = self.gate(torch.cat([h_recon, h_meta], dim=1)) # (B, 128)
        h_fused = g * h_recon + (1.0 - g) * h_meta              # (B, 128)

        return self.head_cls(h_fused)                            # (B, n_classes)

    # ── Pretrained weight loading ──────────────────────────────────────────────

    def load_ecg_backbone(self, ckpt_path: str, device: str = "cpu") -> None:
        """Load pretrained ResNet1dWang weights into the ECG backbone.

        Loads stem, layer1/2/3, head_feat weights.
        Skips head_cls keys that don't match (e.g. different n_classes).
        meta_encoder and gate are always randomly initialized.

        Args:
            ckpt_path: path to ResNet1dWang .pt checkpoint
            device:    map_location device string
        """
        sd = torch.load(ckpt_path, map_location=device)
        own_sd = self.state_dict()

        loaded, skipped = [], []
        for k, v in sd.items():
            if k in own_sd and own_sd[k].shape == v.shape:
                own_sd[k] = v
                loaded.append(k)
            else:
                skipped.append(k)

        self.load_state_dict(own_sd)
        print(f"[load_ecg_backbone] loaded {len(loaded)} / {len(sd)} keys")
        if skipped:
            print(f"  skipped: {skipped}")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = ResNet1dWangGated(n_leads=12, n_classes=5, meta_in_dim=3).to(device)

    B = 4
    x    = torch.randn(B, 12, 1000).to(device)
    meta = torch.randn(B, 3).to(device)

    logits = model(x, meta)
    print(f"input  x   : {tuple(x.shape)}")
    print(f"input  meta: {tuple(meta.shape)}")
    print(f"output     : {tuple(logits.shape)}")   # (4, 5)

    probs = torch.sigmoid(logits)
    print(f"probs  min={probs.min():.4f}  max={probs.max():.4f}")
    print("OK")
