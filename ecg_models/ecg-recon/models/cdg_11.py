# -*- coding: utf-8 -*-
"""
CDGS v11 (`cdg_11.py`): CDGS10 + collapse 방지 정규화
====================================================
목표:
  - 모델 구조는 CDGS10과 동일하게 유지 (최소 변경)
  - Loss에만 아래 2개 항을 추가
    1) envelope decorrelation loss
    2) direction diversity loss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .cdg_10 import CDGS10, CDGS10Loss, get_param_groups


def _select_topk_slots(x: torch.Tensor, amplitude: torch.Tensor, top_k: int) -> torch.Tensor:
    """x의 slot 축(dim=1)에서 amplitude 상위 K개만 선택."""
    if top_k <= 0 or x.size(1) <= top_k:
        return x

    idx = torch.topk(amplitude, k=top_k, dim=1).indices  # [B, K]
    if x.ndim == 3:  # [B, N, T]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, x.size(2))
    elif x.ndim == 4:  # [B, N, C, T]
        gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
    else:
        raise ValueError("Unsupported tensor rank for top-k slot selection")
    return x.gather(dim=1, index=gather_idx)


def _offdiag_abs_mean(sim: torch.Tensor) -> torch.Tensor:
    """pairwise similarity 행렬(sim)의 off-diagonal |cos| 평균."""
    n = sim.size(-1)
    if n <= 1:
        return sim.new_tensor(0.0)

    diag_sum = sim.diagonal(dim1=-2, dim2=-1).abs().sum(dim=-1)
    full_sum = sim.abs().sum(dim=(-2, -1))
    denom = float(n * (n - 1))
    return ((full_sum - diag_sum) / denom).mean()


class CDGS11(CDGS10):
    """구조는 CDGS10과 동일. 차이는 loss에서만 적용."""


class CDGS11Loss(CDGS10Loss):
    def __init__(
        self,
        w_env_decor: float = 0.01,
        w_dir_div: float = 0.005,
        decor_top_k: int = 64,
        env_time_stride: int = 4,
        dir_time_stride: int = 8,
    ):
        super().__init__()
        self.w_env_decor = float(w_env_decor)
        self.w_dir_div = float(w_dir_div)
        self.decor_top_k = int(decor_top_k)
        self.env_time_stride = max(1, int(env_time_stride))
        self.dir_time_stride = max(1, int(dir_time_stride))

    def _envelope_decorrelation(self, envelope_t: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        # [B, N, T] -> 시간 축 다운샘플 후 top-k 슬롯 선택
        env = envelope_t[:, :, :: self.env_time_stride]
        env = _select_topk_slots(env, amplitude, self.decor_top_k)

        # slot별 평균 제거 후 cosine similarity
        env = env - env.mean(dim=-1, keepdim=True)
        env = F.normalize(env, dim=-1, eps=1e-6)
        sim = torch.bmm(env, env.transpose(1, 2))  # [B, K, K]
        return _offdiag_abs_mean(sim)

    def _direction_diversity(self, dipole_dir_t: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        # [B, N, 3, T] -> top-k 슬롯 + 시간 다운샘플
        dirs = dipole_dir_t[:, :, :, :: self.dir_time_stride]
        dirs = _select_topk_slots(dirs, amplitude, self.decor_top_k)
        dirs = F.normalize(dirs, dim=2, eps=1e-6)

        # time별 pairwise cosine을 평균
        b, n, _, t = dirs.shape
        dirs_bt = dirs.permute(0, 3, 1, 2).reshape(b * t, n, 3)
        sim = torch.bmm(dirs_bt, dirs_bt.transpose(1, 2))  # [B*T, K, K]
        return _offdiag_abs_mean(sim)

    def forward(
        self,
        pred,
        target,
        amplitude,
        position=None,
        phys_out=None,
        envelope_t=None,
        dipole_dir_t=None,
        **kwargs,
    ):
        total, terms = super().forward(
            pred,
            target,
            amplitude,
            position=position,
            phys_out=phys_out,
            envelope_t=envelope_t,
            **kwargs,
        )

        zero = pred.new_tensor(0.0)
        l_env_decor = zero
        l_dir_div = zero

        if self.w_env_decor > 0 and envelope_t is not None:
            l_env_decor = self._envelope_decorrelation(envelope_t, amplitude)

        if self.w_dir_div > 0 and dipole_dir_t is not None:
            l_dir_div = self._direction_diversity(dipole_dir_t, amplitude)

        total = total + self.w_env_decor * l_env_decor + self.w_dir_div * l_dir_div

        terms.update(
            {
                "env_decor": l_env_decor,
                "dir_div": l_dir_div,
                "total": total,
            }
        )
        return total, terms
