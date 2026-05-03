#!/usr/bin/env python
"""
Train PTB-XL 1→12 lead ECG reconstruction (v9.4 연동 버전)
==================================================================
[v9.4 수정사항 대응]
  - CDGS9Loss 초기화 파라미터 업데이트 (새로운 물리 제약 가중치들)
  - train_one_epoch / validate 에 lead_id 생성 및 전달
  - region_logits 파라미터 전달 추가
"""

import argparse
import copy
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys, os
from pathlib import Path

# Docker/SSH 원격 환경과 로컬 Windows 환경의 폴더 구조 차이를 모두 지원하기 위한 절대 경로 3중 추가
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))                                 # 1. /workspace/models
sys.path.insert(0, str(base_dir / "ecg_models"))                  # 2. /workspace/ecg_models/models
sys.path.insert(0, str(base_dir / "ecg_models" / "ecg-recon"))    # 3. /workspace/ecg_models/ecg-recon/models
from models import (
    UNet_Hawkiyc, Generator_Zehui, Discriminator_Zehui, VAE_Zehui, LSTM_Zehui,
    ECGrecover_UMMISCO, MCMA_UNet3Plus, UNet1D_Baseline, Diffusion1to12,
    BeatDiff1to12,
    CDGS, CDGS2, CDGS3, CDGS4, CDGS5, CDGS6, CDGS7, CDGS8, CDGS9, CDGS10, CDGS11,
    CDGS12, CDGS13,
    CDGSLoss, CDGS2Loss, CDGS3Loss, CDGS4Loss, CDGS5Loss,
    CDGS6Loss, CDGS7Loss, CDGS8Loss, CDGS9Loss, CDGS10Loss, CDGS11Loss,
    CDGS12Loss, CDGS13Loss,
)
from models.cdg_8 import get_param_groups
from models.cdg_9 import get_param_groups as get_param_groups_9  
from models.cdg_10 import get_param_groups as get_param_groups_10
from models.cdg_11 import get_param_groups as get_param_groups_11
from models.cdg_12 import get_param_groups as get_param_groups_12
from models.cdg_13 import get_param_groups as get_param_groups_13

LEADS    = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
META_DIM = 6

MODEL_REGISTRY = {
    "unet_hawkiyc":    lambda: UNet_Hawkiyc(in_ch=1, out_ch=12, droprate=0.05),
    "unet1d_baseline": lambda: UNet1D_Baseline(in_ch=1, out_ch=12, base_ch=32),
    "diffusion_1to12": lambda: Diffusion1to12(cond_ch=1, out_ch=12, base_ch=64, timesteps=200, sample_steps=30, val_sample_steps=20),
    "beatdiff_1to12":  lambda: BeatDiff1to12(cond_ch=1, out_ch=12, base_ch=192, channel_mult=(1,2,3,4), num_blocks=3, sample_steps=50, val_sample_steps=35),
    "cdg_8":           lambda: CDGS8(d_model=128, n_gaussians=1024, n_encoder_layers=4),
    "cdg_9":           lambda: CDGS9(d_model=256, n_gaussians=512,  n_encoder_layers=4),
    "cdg_10":          lambda: CDGS10(d_model=256, n_gaussians=512, n_encoder_layers=4),
    "cdg_11":          lambda: CDGS11(d_model=256, n_gaussians=512, n_encoder_layers=4),
    "cdg_12":          lambda: CDGS12(d_model=256, n_gaussians=32, n_encoder_layers=4, n_temporal_bases=8),
    "cdg_13":          lambda: CDGS13(d_model=256, n_gaussians=64, n_encoder_layers=4, n_temporal_bases=16),
}


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_ptbxl_downloaded(data_dir):
    if (data_dir / "ptbxl_database.csv").exists(): return
    data_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("ptb-xl", str(data_dir))

def get_grad_norm(module):
    return sum(p.grad.data.norm(2).item()**2 for p in module.parameters() if p.grad is not None) ** 0.5


class EMA:
    """Exponential Moving Average for diffusion model weights."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)

def build_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.05):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def get_bypass_alpha(epoch, warmup=5, full=20, max_a=1.0):
    """[v9.5] warmup 단축: bypass를 일찍 켜서 gate가 빨리 학습 시작하도록"""
    if epoch <= warmup: return 0.0
    if epoch >= full:   return max_a
    return max_a * (epoch - warmup) / (full - warmup)


# ── Metadata ──────────────────────────────────────────────────────────────────
def build_meta_lookup(df, file_col):
    valid_h   = pd.to_numeric(df.get("height", pd.Series(dtype=float)), errors='coerce')
    valid_w   = pd.to_numeric(df.get("weight", pd.Series(dtype=float)), errors='coerce')
    valid_age = pd.to_numeric(df.get("age",    pd.Series(dtype=float)), errors='coerce')
    valid_sex = pd.to_numeric(df.get("sex",    pd.Series(dtype=float)), errors='coerce')

    df_tmp = df.copy()
    df_tmp[["_age","_sex","_h","_w"]] = pd.DataFrame({
        "_age": valid_age, "_sex": valid_sex, "_h": valid_h, "_w": valid_w
    })
    df_tmp["_age_group"] = pd.cut(df_tmp["_age"], bins=[0,30,50,70,120], labels=[0,1,2,3])
    group_means = df_tmp.groupby(["_sex","_age_group"], observed=True)[["_h","_w"]].mean()
    global_h = valid_h.median() if not valid_h.isna().all() else 165.0
    global_w = valid_w.median() if not valid_w.isna().all() else  65.0

    def grp_mean(sex, age, col):
        try:
            ag = pd.cut([age], bins=[0,30,50,70,120], labels=[0,1,2,3])[0]
            v  = group_means.loc[(sex, ag), col]
            if not np.isnan(v): return float(v)
        except: pass
        return global_h if col == "_h" else global_w

    lookup = {}
    for _, row in df.iterrows():
        fname = row[file_col]
        try:    age_r = float(row.get("age", 50)); age_r = 50.0 if np.isnan(age_r) else age_r
        except: age_r = 50.0
        age_n = float(np.clip(age_r / 100.0, 0.0, 1.0))

        try:    sex = int(row.get("sex", 2)); sex = 2 if sex not in (0,1) else sex
        except: sex = 2

        h_miss = 0.0
        try:
            h = float(row.get("height", float("nan")))
            if np.isnan(h): h = grp_mean(sex if sex in (0,1) else 0, age_r, "_h"); h_miss = 1.0
        except: h = global_h; h_miss = 1.0
        h_n = float((h - 165.0) / 15.0)

        w_miss = 0.0
        try:
            w = float(row.get("weight", float("nan")))
            if np.isnan(w): w = grp_mean(sex if sex in (0,1) else 0, age_r, "_w"); w_miss = 1.0
        except: w = global_w; w_miss = 1.0
        w_n = float((w - 65.0) / 15.0)

        lookup[fname] = (age_n, sex, h_n, w_n, h_miss, w_miss)
    return lookup


# ── Dataset ───────────────────────────────────────────────────────────────────
class PTBXLReconstructionDataset(Dataset):
    def __init__(self, data_dir, record_paths, input_idx, meta_lookup):
        self.data_dir = data_dir; self.record_paths = record_paths
        self.input_idx = input_idx; self.meta_lookup = meta_lookup

    def __len__(self): return len(self.record_paths)

    def __getitem__(self, idx):
        rec    = self.record_paths[idx]
        sig, _ = wfdb.rdsamp(str(self.data_dir / rec))
        sig    = sig.astype(np.float32)
        sig    = (sig - sig.mean(0, keepdims=True)) / (sig.std(0, keepdims=True) + 1e-6)
        sig    = sig.T
        x      = sig[self.input_idx:self.input_idx+1]
        meta   = torch.tensor(list(self.meta_lookup.get(rec, (0.5,2,0.,0.,1.,1.))), dtype=torch.float32)
        return torch.from_numpy(x), torch.from_numpy(sig), meta


# ── 평가 함수 ─────────────────────────────────────────────────────────────────
def _mask(n, idx, device):
    m = torch.ones(n, dtype=torch.bool, device=device); m[idx] = False; return m

def mae_no_input(pred, tgt, idx):
    m = _mask(pred.size(1), idx, pred.device)
    return F.l1_loss(pred[:, m], tgt[:, m]).item()

def pearson_no_input(pred, tgt, idx):
    p, t = pred.detach().cpu().numpy(), tgt.detach().cpu().numpy()
    corrs = []
    for li in range(p.shape[1]):
        if li == idx: continue
        pv, tv = p[:, li].reshape(-1), t[:, li].reshape(-1)
        if np.std(pv) < 1e-8 or np.std(tv) < 1e-8: continue
        c = float(np.corrcoef(pv, tv)[0,1])
        if not np.isnan(c): corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0

def recon_loss(pred, tgt, idx):
    m = _mask(pred.size(1), idx, pred.device)
    return 0.3 * F.l1_loss(pred, tgt) + 0.7 * F.l1_loss(pred[:, m], tgt[:, m])


# ── Train / Validate ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, device, input_idx,
                    model_name, epoch, loss_fn, bypass_alpha):
    model.train()
    run_loss = run_mae = 0.0
    dbg = {"vcg_m":0., "env":0., "amp":0., "gate":0., "steps":0}
    # [v10] cdg_10~13은 1/r³ 물리 연산이 fp16 범위 초과 → AMP 비활성화
    use_amp = device.type == "cuda" and model_name not in ("cdg_8", "cdg_10", "cdg_11", "cdg_12", "cdg_13", "beatdiff_1to12")

    for x, y, meta in tqdm(loader, desc="train", leave=False):
        x, y, meta = x.to(device), y.to(device), meta.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if model_name in ("diffusion_1to12", "beatdiff_1to12"):
                actual_m = model.module if hasattr(model, "module") else model
                loss, pred, loss_d = actual_m.loss_and_pred(x, y)
            elif model_name == "cdg_13":
                actual_m = model.module if hasattr(model, "module") else model
                stage = actual_m.stage
                lead_id = torch.full((x.size(0),), input_idx, dtype=torch.long, device=x.device)
                
                if stage == 1:
                    pred, phys_out, extras = model(y, meta=meta, bypass_alpha=bypass_alpha)
                    loss, loss_d = loss_fn(
                        pred, y, extras["amplitude"], stage=stage,
                        phys_out=phys_out, temporal_bases=extras.get("temporal_bases"),
                        dipole_dir=extras.get("dipole_dir"), envelope_t=extras.get("envelope_t"),
                    )
                else:
                    pred, phys_out, extras = model(x, meta=meta, bypass_alpha=bypass_alpha, lead_id=lead_id, x_12=y)
                    loss, loss_d = loss_fn(
                        pred, y, extras["amplitude"], stage=stage,
                        phys_out=phys_out, temporal_bases=extras.get("temporal_bases"),
                        dipole_dir=extras.get("dipole_dir"), envelope_t=extras.get("envelope_t"),
                        extras=extras,
                    )
                dbg["env"]   += extras["envelope_t"].mean().item()
                dbg["amp"]   += extras["amplitude"].mean().item()
                dbg["gate"]  += extras["gate"].mean().item()
                if "distill" in loss_d and torch.is_tensor(loss_d["distill"]) and loss_d["distill"].item() > 0:
                    dbg["distill"] = dbg.get("distill", 0.) + loss_d["distill"].item()
                dbg["steps"] += 1

            elif model_name == "cdg_12":
                lead_id = torch.full((x.size(0),), input_idx, dtype=torch.long, device=x.device)
                pred, phys_out, extras = model(x, meta=meta, bypass_alpha=bypass_alpha, lead_id=lead_id)

                loss, loss_d = loss_fn(
                    pred, y,
                    extras["amplitude"],
                    phys_out=phys_out,
                    temporal_bases=extras.get("temporal_bases"),
                    dipole_dir=extras.get("dipole_dir"),
                    envelope_t=extras.get("envelope_t"),
                )
                dbg["env"]   += extras["envelope_t"].mean().item()
                dbg["amp"]   += extras["amplitude"].mean().item()
                dbg["gate"]  += extras["gate"].mean().item()
                dbg["steps"] += 1

            elif model_name in ("cdg_10", "cdg_11"):
                lead_id = torch.full((x.size(0),), input_idx, dtype=torch.long, device=x.device)
                pred, phys_out, extras = model(x, meta=meta, bypass_alpha=bypass_alpha, lead_id=lead_id)

                loss, loss_d = loss_fn(
                    pred, y,
                    extras["amplitude"],
                    position=extras.get("position"),
                    phys_out=phys_out,
                    envelope_t=extras.get("envelope_t"),
                    dipole_dir_t=extras.get("dipole_dir_t"),
                )
                dbg["env"]   += extras["envelope_t"].mean().item()
                dbg["amp"]   += extras["amplitude"].mean().item()
                dbg["gate"]  += extras["gate"].mean().item()
                # 위치 통계
                pos = extras["position"]
                dbg["pos_std"] = dbg.get("pos_std", 0.) + pos.std().item()
                dbg["steps"] += 1

            elif model_name == "cdg_9":
                # lead_id 주입
                lead_id = torch.full((x.size(0),), input_idx, dtype=torch.long, device=x.device)
                pred, _, extras = model(x, meta=meta, bypass_alpha=bypass_alpha, lead_id=lead_id)
                
                loss, loss_d = loss_fn(
                    pred, y,
                    extras["amplitude"], extras["envelope_t"],
                    vcg=extras.get("vcg"), gate=extras.get("gate"),
                    region_logits=extras.get("region_logits")
                )
                dbg["vcg_m"] += extras["vcg"].norm(dim=1).mean().item() if "vcg" in extras else 0.0
                dbg["env"]   += extras["envelope_t"].mean().item()
                dbg["amp"]   += extras["amplitude"].mean().item()
                dbg["gate"]  += extras["gate"].mean().item() if "gate" in extras else 0.0
                dbg["steps"] += 1

            elif model_name == "cdg_8":
                pred, _, extras = model(x, meta=meta, bypass_alpha=bypass_alpha)
                amp = extras.get("amplitude", torch.zeros(1, device=pred.device, dtype=pred.dtype))
                loss, _ = loss_fn(
                    pred, pred, y, amplitude=amp,
                    envelope_t=extras.get("envelope_t"),
                    p_pos=extras.get("p_pos"),
                    gate=extras.get("physics_gate"),
                )
            else:
                pred = model(x)
                loss = recon_loss(pred, y, input_idx)

        # NaN loss 감지 → skip (가중치 오염 방지)
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        if model_name == "cdg_8":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # cdg_10/11: position gradient를 별도로 강하게 clip (1/r³ 안정화)
            if model_name in ("cdg_10", "cdg_11"):  # cdg_12 제외: position이 buffer라 grad 없음
                actual_m = model.module if hasattr(model, "module") else model
                if hasattr(actual_m, "predictor") and hasattr(actual_m.predictor, "positions"):
                    if actual_m.predictor.positions.requires_grad:
                        torch.nn.utils.clip_grad_norm_([actual_m.predictor.positions], max_norm=0.1)
            # cdg_13 Stage 1: position이 nn.Parameter → 강한 clip
            if model_name == "cdg_13":
                actual_m = model.module if hasattr(model, "module") else model
                if hasattr(actual_m, "predictor") and actual_m.predictor.positions.requires_grad:
                    torch.nn.utils.clip_grad_norm_([actual_m.predictor.positions], max_norm=0.1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()

        run_loss += loss.item() * x.size(0)
        run_mae  += mae_no_input(pred, y, input_idx) * x.size(0)

    n = len(loader.dataset)
    metrics = {"loss": run_loss/n, "mae_no_input": run_mae/n}
    if model_name in ("cdg_9", "cdg_10", "cdg_11", "cdg_12", "cdg_13") and dbg["steps"] > 0:
        s = dbg["steps"]
        metrics["debug"] = {k: v/s for k,v in dbg.items() if k!="steps"}
    return metrics


@torch.no_grad()
def validate(model, loader, device, input_idx, model_name, loss_fn):
    model.eval()
    run_loss = run_mae = run_corr = 0.0

    for x, y, meta in tqdm(loader, desc="valid", leave=False):
        x, y, meta = x.to(device), y.to(device), meta.to(device)

        if model_name in ("diffusion_1to12", "beatdiff_1to12"):
            actual_m = model.module if hasattr(model, "module") else model
            pred = actual_m.sample(x, steps=actual_m.val_sample_steps)
            loss = recon_loss(pred, y, input_idx)
        elif model_name == "cdg_13":
            actual_m = model.module if hasattr(model, "module") else model
            stage = actual_m.stage
            lead_id = torch.full((x.size(0),), input_idx, dtype=torch.long, device=x.device)
            if stage == 1:
                pred, _, extras = model(y, meta=meta, bypass_alpha=1.0)
            else:
                pred, _, extras = model(x, meta=meta, bypass_alpha=1.0, lead_id=lead_id)
            
            loss, _ = loss_fn(
                pred, y, extras["amplitude"], stage=stage,
                temporal_bases=extras.get("temporal_bases"),
                dipole_dir=extras.get("dipole_dir"), envelope_t=extras.get("envelope_t"),
            )
        elif model_name == "cdg_12":
            lead_id = torch.full((x.size(0),), input_idx, dtype=torch.long, device=x.device)
            pred, _, extras = model(x, meta=meta, bypass_alpha=1.0, lead_id=lead_id)
            
            loss, _ = loss_fn(
                pred, y,
                extras["amplitude"],
                temporal_bases=extras.get("temporal_bases"),
                dipole_dir=extras.get("dipole_dir"),
                envelope_t=extras.get("envelope_t"),
            )
        elif model_name in ("cdg_10", "cdg_11"):
            lead_id = torch.full((x.size(0),), input_idx, dtype=torch.long, device=x.device)
            pred, _, extras = model(x, meta=meta, bypass_alpha=1.0, lead_id=lead_id)
            
            loss, _ = loss_fn(
                pred, y,
                extras["amplitude"],
                position=extras.get("position"),
                envelope_t=extras.get("envelope_t"),
                dipole_dir_t=extras.get("dipole_dir_t"),
            )
        elif model_name == "cdg_9":
            lead_id = torch.full((x.size(0),), input_idx, dtype=torch.long, device=x.device)
            pred, _, extras = model(x, meta=meta, bypass_alpha=1.0, lead_id=lead_id)
            
            loss, _ = loss_fn(
                pred, y,
                extras["amplitude"], extras["envelope_t"],
                vcg=extras.get("vcg"), gate=extras.get("gate"),
                region_logits=extras.get("region_logits")
            )
        elif model_name == "cdg_8":
            pred, _, extras = model(x, meta=meta, bypass_alpha=1.0)
            amp = extras.get("amplitude", torch.zeros(1, device=pred.device, dtype=pred.dtype))
            loss, _ = loss_fn(
                pred, pred, y, amplitude=amp,
                envelope_t=extras.get("envelope_t"),
                p_pos=extras.get("p_pos"),
                gate=extras.get("physics_gate"),
            )
        else:
            pred = model(x)
            loss = recon_loss(pred, y, input_idx)

        run_loss += loss.item() * x.size(0)
        run_mae  += mae_no_input(pred, y, input_idx) * x.size(0)
        run_corr += pearson_no_input(pred, y, input_idx) * x.size(0)

    n = len(loader.dataset)
    return {"loss": run_loss/n, "mae_no_input": run_mae/n, "pearson_no_input": run_corr/n}


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         type=str,   default="cdg_10")
    p.add_argument("--data_dir",      type=str,   default="./data/ptb-xl")
    p.add_argument("--output_dir",    type=str,   default="./outputs")
    p.add_argument("--input_lead",    type=str,   default="I", choices=LEADS)
    p.add_argument("--use_hr",        action="store_true")
    p.add_argument("--val_folds",     type=str,   default="9")
    p.add_argument("--test_folds",    type=str,   default="10")
    p.add_argument("--max_records",   type=int,   default=0)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--lr",            type=float, default=2e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--min_lr_ratio",  type=float, default=0.05)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--skip_download", action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    seed_everything(args.seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download: ensure_ptbxl_downloaded(data_dir)
    df       = pd.read_csv(data_dir / "ptbxl_database.csv")
    file_col = "filename_hr" if args.use_hr else "filename_lr"
    meta_lut = build_meta_lookup(df, file_col)

    val_folds  = {int(x) for x in args.val_folds.split(",")  if x.strip()}
    test_folds = {int(x) for x in args.test_folds.split(",") if x.strip()}
    train_recs = df[~df["strat_fold"].isin(val_folds|test_folds)][file_col].tolist()
    val_recs   = df[df["strat_fold"].isin(val_folds)][file_col].tolist()

    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_recs); rng.shuffle(val_recs)
    if args.max_records > 0:
        t = max(1, int(args.max_records * 0.8))
        train_recs, val_recs = train_recs[:t], val_recs[:args.max_records-t]

    idx      = LEADS.index(args.input_lead)
    train_ds = PTBXLReconstructionDataset(data_dir, train_recs, idx, meta_lut)
    val_ds   = PTBXLReconstructionDataset(data_dir, val_recs,   idx, meta_lut)
    kw       = dict(num_workers=args.num_workers, pin_memory=True)
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,  **kw)
    val_ld   = DataLoader(val_ds,   args.batch_size, shuffle=False, **kw)

    # ── 모델 & Loss ─────────────────────────────────────────────────────────
    model   = MODEL_REGISTRY[args.model]().to(device)
    if torch.cuda.device_count() > 1 and args.model not in ("diffusion_1to12", "beatdiff_1to12"):
        model = nn.DataParallel(model)

    if args.model == "cdg_13":
        loss_fn = CDGS13Loss()
    elif args.model == "cdg_12":
        loss_fn = CDGS12Loss()
    elif args.model == "cdg_11":
        loss_fn = CDGS11Loss()
    elif args.model == "cdg_10":
        loss_fn = CDGS10Loss()
    elif args.model == "cdg_9":
        # v9.4에서 제거/추가된 가우시안 제약(gate 제거 등)을 적용하기 위해 빈 인자로 초기화 (기본값이 최적)
        loss_fn = CDGS9Loss()
    elif args.model == "cdg_8":
        from models.cdg_8 import CDGS8Loss
        loss_fn = CDGS8Loss()
    else:
        loss_fn = None

    # ── Optimizer ───────────────────────────────────────────────────────────
    # [v9.5] cdg_9는 lr 그대로 사용 (2e-3 × 2 = 4e-3는 너무 높음)
    if args.model == "cdg_13":
        train_lr = 1e-4  # Stage 1: learnable positions + physics → 낮은 lr 필요
    elif args.model in ("cdg_10", "cdg_11", "cdg_12"):
        train_lr = args.lr
    elif args.model == "cdg_9":
        train_lr = args.lr  # 기본 2e-3
    elif args.model == "cdg_8":
        train_lr = args.lr * 2.0
    elif args.model == "beatdiff_1to12":
        train_lr = 3e-4  # diffusion 모델은 낮은 lr 필요
    else:
        train_lr = args.lr
    actual    = model.module if hasattr(model, "module") else model
    if args.model == "cdg_13":
        pgs = get_param_groups_13(actual, args.weight_decay)
        for pg in pgs: pg["lr"] = train_lr
        optimizer = torch.optim.AdamW(pgs)
    elif args.model == "cdg_12":
        pgs = get_param_groups_12(actual, args.weight_decay)
        for pg in pgs:
            pg["lr"] = train_lr
        optimizer = torch.optim.AdamW(pgs)
    elif args.model in ("cdg_10", "cdg_11"):
        pg_fn = get_param_groups_11 if args.model == "cdg_11" else get_param_groups_10
        pgs = pg_fn(actual, args.weight_decay)
        for pg in pgs:
            scale = pg.pop("lr_scale", 1.0)  # position 그룹은 0.05
            pg["lr"] = train_lr * scale
        optimizer = torch.optim.AdamW(pgs)
    elif args.model == "cdg_9":
        pgs = get_param_groups_9(actual, args.weight_decay)
        for pg in pgs: pg["lr"] = train_lr
        optimizer = torch.optim.AdamW(pgs)
    elif args.model == "cdg_8":
        pgs = get_param_groups(actual, args.weight_decay)
        for pg in pgs: pg["lr"] = train_lr
        optimizer = torch.optim.AdamW(pgs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_lr, weight_decay=args.weight_decay)

    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    # CDG13: warmup 없이 바로 full lr에서 cosine decay
    warmup_ep = 0 if args.model == "cdg_13" else args.warmup_epochs
    scheduler = build_scheduler(optimizer, warmup_ep, args.epochs, args.min_lr_ratio)

    # EMA for diffusion models
    ema = None
    if args.model == "beatdiff_1to12":
        ema = EMA(actual, decay=0.9999)

    best_val = float("inf")

    for epoch in range(1, args.epochs+1):
        if args.model == "cdg_13":
            actual_m = model.module if hasattr(model, "module") else model
            # CDG13 2-stage 전환 로직
            mid_epoch = args.epochs // 2 + 1
            if epoch == mid_epoch:
                actual_m.set_stage(2)
                # Stage 2: random init encoder_1 → 더 낮은 lr로 안정적 학습
                stage2_lr = 1e-4
                pgs = get_param_groups_13(actual_m, args.weight_decay)
                for pg in pgs: pg["lr"] = stage2_lr
                optimizer = torch.optim.AdamW(pgs)
                remaining = args.epochs - args.epochs // 2
                scheduler = build_scheduler(optimizer, 0, remaining, args.min_lr_ratio)  # warmup 없이 바로 시작
                print(f"  [CDG13] Stage 2 lr={stage2_lr:.1e}, remaining={remaining} epochs")
            bypass_alpha = get_bypass_alpha(epoch - (mid_epoch if actual_m.stage == 2 else 0), warmup=10, full=20)
        elif args.model in ("cdg_10", "cdg_11", "cdg_12"):
            # epoch 30부터 bypass 도입, 이후 완만히 1.0까지 증가
            bypass_alpha = get_bypass_alpha(epoch, warmup=29, full=45)
        else:
            bypass_alpha = get_bypass_alpha(epoch) if args.model in ("cdg_8","cdg_9") else 1.0
        cur_lr       = optimizer.param_groups[0]["lr"]

        tr = train_one_epoch(model, train_ld, optimizer, scaler, device, idx,
                             args.model, epoch, loss_fn, bypass_alpha)

        # EMA update after each training epoch
        if ema is not None:
            ema.update(actual)

        # BeatDiff validation: use EMA weights for better quality
        if ema is not None:
            vl = validate(ema.shadow, val_ld, device, idx, args.model, loss_fn)
        else:
            vl = validate(model, val_ld, device, idx, args.model, loss_fn)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"lr={cur_lr:.2e} bypass={bypass_alpha:.2f} | "
            f"train={tr['loss']:.4f}  val_mae={vl['mae_no_input']:.4f}  "
            f"val_pearson={vl['pearson_no_input']:.4f}"
        )
        if "debug" in tr:
            d = tr["debug"]
            if args.model == "cdg_13":
                actual_m = model.module if hasattr(model, "module") else model
                dist_str = f"| Distill:{d['distill']:.3f}" if "distill" in d else ""
                print(
                    f"  \u2514\u2500 [{args.model} Stg{actual_m.stage}] Env:{d['env']:.3f} | Amp:{d['amp']:.3f} | "
                    f"Gate(Phys%):{d['gate']:.3f} {dist_str}"
                )
            elif args.model == "cdg_12":
                print(
                    f"  \u2514\u2500 [{args.model}] Env:{d['env']:.3f} | Amp:{d['amp']:.3f} | "
                    f"Gate(Phys%):{d['gate']:.3f} | Pos:FIXED"
                )
            elif args.model in ("cdg_10", "cdg_11"):
                print(
                    f"  \u2514\u2500 [{args.model}] Env:{d['env']:.3f} | Amp:{d['amp']:.3f} | "
                    f"Gate(Phys%):{d['gate']:.3f} | Pos_std:{d.get('pos_std',0):.4f}"
                )
            else:
                vcg_warn = " ⚠ VCG 폭발!" if d["vcg_m"] > 3.0 else (" ✓" if d["vcg_m"] < 2.0 else "")
                print(
                    f"  └─ [cdg_9] VCG_Mag:{d['vcg_m']:.3f}{vcg_warn} | "
                    f"Env:{d['env']:.3f} | Amp:{d['amp']:.3f} | Gate(Phys%):{d['gate']:.3f}"
                )

        # Save checkpoint (with EMA state for diffusion models)
        ckpt = {"epoch": epoch, "model_state": model.state_dict()}
        if ema is not None:
            ckpt["ema_state"] = ema.state_dict()
        torch.save(ckpt, out_dir/"last.pt")
        if vl["mae_no_input"] < best_val:
            best_val = vl["mae_no_input"]
            torch.save(ckpt, out_dir/"best.pt")
            print(f"  └─ ★ best (val_mae={best_val:.4f})")


if __name__ == "__main__":
    main()