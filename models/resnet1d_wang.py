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
        ], dim=1).squeeze(-1)  # → (B, 2*C)


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
            x: (B, n_leads, T)  — float32, z-score normalized per lead
        Returns:
            logits: (B, n_classes)  — apply sigmoid for probabilities
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = ResNet1dWang(n_leads=12, n_classes=5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet1dWang  params: {n_params/1e6:.2f}M")

    x = torch.randn(4, 12, 1000)
    out = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")   # (4, 5)
