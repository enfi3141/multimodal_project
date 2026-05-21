"""Cascade lead generator: I → II → limb (arithmetic + residual) → V leads
   + Teacher forcing + Metadata FiLM conditioning.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
IDX = {l: i for i, l in enumerate(LEADS)}


class ConvBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ic, oc, 7, padding=3), nn.BatchNorm1d(oc), nn.GELU(),
            nn.Conv1d(oc, oc, 5, padding=2), nn.BatchNorm1d(oc), nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.down = nn.Conv1d(ic, oc, 4, stride=2, padding=1)
        self.block = ConvBlock(oc, oc)
    def forward(self, x): return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.up = nn.ConvTranspose1d(ic, oc, 4, stride=2, padding=1)
        self.block = ConvBlock(oc + sc, oc)
    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class UNet1D(nn.Module):
    """UNet with optional FiLM conditioning at bottleneck."""
    def __init__(self, in_ch, base_ch=32, out_ch=1, film_dim=0):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = DownBlock(base_ch, base_ch * 2)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)
        self.bot = ConvBlock(base_ch * 8, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)
        self.head = nn.Conv1d(base_ch, out_ch, 1)
        self.film = nn.Linear(film_dim, base_ch * 8 * 2) if film_dim > 0 else None

    def forward(self, x, meta_h=None):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        b = self.bot(e4)
        if self.film is not None and meta_h is not None:
            gb = self.film(meta_h)
            gamma, beta = gb.chunk(2, dim=-1)
            b = b * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        d3 = self.up3(b, e3); d2 = self.up2(d3, e2); d1 = self.up1(d2, e1)
        return self.head(d1)


class CascadeECGRecon(nn.Module):
    def __init__(self, base_ch=32, res_base_ch=16, residual_scale=0.3, meta_dim=0):
        super().__init__()
        self.residual_scale = residual_scale
        self.meta_dim = meta_dim
        film_d = 64 if meta_dim > 0 else 0

        self.meta_enc = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.GELU(),
            nn.Linear(32, 64),
        ) if meta_dim > 0 else None

        self.gen_II = UNet1D(1, base_ch, 1, film_dim=film_d)
        self.res_III = UNet1D(2, res_base_ch, 1)
        self.res_aVR = UNet1D(2, res_base_ch, 1)
        self.res_aVL = UNet1D(2, res_base_ch, 1)
        self.res_aVF = UNet1D(2, res_base_ch, 1)
        self.gen_V = UNet1D(6, base_ch, 6, film_dim=film_d)

    def forward(self, x_I, gt_full=None, teacher_forcing=0.0,
                meta=None, return_intermediate=False):
        meta_h = self.meta_enc(meta) if (self.meta_enc is not None and meta is not None) else None

        pred_II = self.gen_II(x_I, meta_h=meta_h)

        # Teacher forcing
        use_gt = (gt_full is not None) and (torch.rand(1).item() < teacher_forcing)
        ctx_II = gt_full[:, IDX["II"]:IDX["II"]+1] if use_gt else pred_II

        ctx2 = torch.cat([x_I, ctx_II], dim=1)
        arith_III = ctx_II - x_I
        arith_aVR = -(x_I + ctx_II) / 2.0
        arith_aVL = x_I - ctx_II / 2.0
        arith_aVF = ctx_II - x_I / 2.0

        s = self.residual_scale
        pred_III = arith_III + s * torch.tanh(self.res_III(ctx2))
        pred_aVR = arith_aVR + s * torch.tanh(self.res_aVR(ctx2))
        pred_aVL = arith_aVL + s * torch.tanh(self.res_aVL(ctx2))
        pred_aVF = arith_aVF + s * torch.tanh(self.res_aVF(ctx2))

        if use_gt and gt_full is not None:
            ctx3 = torch.cat([
                x_I,
                gt_full[:, IDX["II"]:IDX["II"]+1],
                gt_full[:, IDX["III"]:IDX["III"]+1],
                gt_full[:, IDX["aVR"]:IDX["aVR"]+1],
                gt_full[:, IDX["aVL"]:IDX["aVL"]+1],
                gt_full[:, IDX["aVF"]:IDX["aVF"]+1],
            ], dim=1)
        else:
            ctx3 = torch.cat([x_I, pred_II, pred_III, pred_aVR, pred_aVL, pred_aVF], dim=1)
        pred_V = self.gen_V(ctx3, meta_h=meta_h)

        out = torch.cat([x_I, pred_II, pred_III, pred_aVR, pred_aVL, pred_aVF, pred_V], dim=1)

        if not return_intermediate:
            return out
        return out, {
            "pred_II": pred_II,
            "pred_III": pred_III, "pred_aVR": pred_aVR,
            "pred_aVL": pred_aVL, "pred_aVF": pred_aVF,
            "pred_V": pred_V,
            "tf_used": use_gt,
        }


class VanillaECGRecon(nn.Module):
    """1-lead → 12-lead direct baseline."""
    def __init__(self, base_ch=32, meta_dim=0):
        super().__init__()
        self.meta_dim = meta_dim
        film_d = 64 if meta_dim > 0 else 0
        self.meta_enc = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.GELU(),
            nn.Linear(32, 64),
        ) if meta_dim > 0 else None
        self.unet = UNet1D(1, base_ch, 12, film_dim=film_d)

    def forward(self, x_I, gt_full=None, teacher_forcing=0.0,
                meta=None, return_intermediate=False):
        meta_h = self.meta_enc(meta) if (self.meta_enc is not None and meta is not None) else None
        out = self.unet(x_I, meta_h=meta_h)
        out_full = out.clone()
        out_full[:, IDX["I"]:IDX["I"]+1] = x_I
        if not return_intermediate:
            return out_full
        return out_full, {"pred_II": out_full[:, IDX["II"]:IDX["II"]+1], "tf_used": False}


# ── Loss ──
LEAD_WEIGHTS = {"II": 2.0, "III": 3.0, "aVF": 3.0, "V2": 2.0, "V4": 2.0, "V5": 2.0}


def get_lead_weight_tensor(device=None):
    w = torch.ones(12, dtype=torch.float32)
    for i, n in enumerate(LEADS):
        w[i] = LEAD_WEIGHTS.get(n, 1.0)
    if device is not None:
        w = w.to(device)
    return w.view(1, 12, 1)


def cascade_loss(pred, target, intermediate=None, lead_w=None, w_II_extra=0.5):
    diff_l1 = (pred - target).abs()
    diff_l2 = (pred - target).pow(2)
    if lead_w is not None:
        diff_l1 = diff_l1 * lead_w
        diff_l2 = diff_l2 * lead_w
    loss = diff_l1.mean() + 0.1 * diff_l2.mean()
    if intermediate is not None and "pred_II" in intermediate and w_II_extra > 0:
        loss = loss + w_II_extra * F.l1_loss(intermediate["pred_II"], target[:, IDX["II"]:IDX["II"]+1])
    return loss


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
