#!/usr/bin/env python
# pyright: reportMissingImports=false
"""
nohup docker run --rm --gpus "device=0" \
  -v ~/project_jyu/physionet.org/files/ptb-xl/1.0.3:/workspace/data/ptb-xl \
  -v "$(pwd)":/workspace \
  -w /workspace \
  tensorflow/tensorflow:2.8.0-gpu \
  bash -lc "pip install -q wfdb pandas scipy tensorflow-addons==0.21.0 && python ecg_models/ecg-recon/inference_all_models.py --data_dir /workspace/data/ptb-xl --output_dir /workspace/ecg_models/ecg-recon/outputs_ptbxl_test --models ekgan,pix2pix --weights ekgan=/workspace/weights/ekgan.weights.h5,pix2pix=/workspace/weights/pix2pix.weights.h5 --input_lead I --test_folds 10 --batch_size 16 --skip_download" \
  > ecg_recon_infer.log 2>&1 < /dev/null &
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_all_models import LEADS, PTBXLReconstructionDataset, build_meta_lookup, seed_everything

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ecg_models"))
from models import (
    CDGS,
    CDGS2,
    CDGS3,
    CDGS4,
    CDGS5,
    CDGS6,
    CDGS7,
    CDGS8,
    CDGS9,
    CDGS10,
    CDGS11,
    ECGrecover_UMMISCO,
    Generator_Zehui,
    LSTM_Zehui,
    MCMA_UNet3Plus,
    Diffusion1to12,
    BeatDiff1to12,
    UNet1D_Baseline,
    UNet_Hawkiyc,
    VAE_Zehui,
)


MODEL_NAMES = [
    "unet_hawkiyc",
    "gan_zehui",
    "vae_zehui",
    "lstm_zehui",
    "ecgrecover",
    "mcma",
    "unet1d_baseline",
    "diffusion_1to12",
    "beatdiff_1to12",
    "cdgs",
    "cdg_2",
    "cdg_3",
    "cdg_4",
    "cdg_5",
    "cdg_6",
    "cdg_7",
    "cdg_8",
    "cdg_9",
    "cdg_10",
    "cdg_11",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Infer and save 1->12 lead ECG")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--data_dir", type=str, default="./data/ptb-xl")
    parser.add_argument("--save_dir", type=str, default="./inference_results")
    parser.add_argument("--input_lead", type=str, default="I", choices=LEADS)
    parser.add_argument("--use_hr", action="store_true")
    parser.add_argument("--test_folds", type=str, default="10")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--acc_tolerance", type=float, default=0.10)

    # CDGS compatibility options (mostly used by cdg_6/cdg_7 variants)
    parser.add_argument("--cdgs_direct_lead_sum", action="store_true")
    parser.add_argument("--cdgs_direct_patch", type=int, default=5)
    parser.add_argument("--cdgs_direct_span", type=float, default=0.55)
    return parser.parse_args()


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("model_state", "state_dict", "model", "model_state_dict"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                state_dict = ckpt_obj[key]
                break
        else:
            state_dict = ckpt_obj
    else:
        state_dict = ckpt_obj

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def infer_arch_hparams(state_dict):
    d_model = None
    n_gaussians = None
    n_encoder_layers = None
    d_deform_hidden = None

    if "gaussian_predictor.gaussian_queries" in state_dict:
        shape = state_dict["gaussian_predictor.gaussian_queries"].shape
        if len(shape) >= 3:
            n_gaussians = int(shape[1])
            d_model = int(shape[2])

    if "predictor.positions" in state_dict:
        p_shape = state_dict["predictor.positions"].shape
        if len(p_shape) >= 2:
            n_gaussians = int(p_shape[0])

    if d_model is None and "encoder.input_proj.weight" in state_dict:
        d_model = int(state_dict["encoder.input_proj.weight"].shape[0])

    if "deform_mlp.net.0.weight" in state_dict:
        d_deform_hidden = int(state_dict["deform_mlp.net.0.weight"].shape[0])

    layer_ids = []
    layer_pat = re.compile(r"^encoder\.transformer_layers\.(\d+)\.")
    for key in state_dict.keys():
        match = layer_pat.match(key)
        if match:
            layer_ids.append(int(match.group(1)))
    if layer_ids:
        n_encoder_layers = max(layer_ids) + 1

    return {
        "d_model": d_model,
        "n_gaussians": n_gaussians,
        "n_encoder_layers": n_encoder_layers,
        "d_deform_hidden": d_deform_hidden,
    }


def build_model(model_name, args, arch_cfg):
    d_model = arch_cfg["d_model"]
    n_gaussians = arch_cfg["n_gaussians"]
    n_encoder_layers = arch_cfg["n_encoder_layers"]
    d_deform_hidden = arch_cfg["d_deform_hidden"]

    if model_name == "unet_hawkiyc":
        return UNet_Hawkiyc(in_ch=1, out_ch=12, droprate=0.05)
    if model_name == "gan_zehui":
        return Generator_Zehui(in_channels=1, out_channels=12)
    if model_name == "vae_zehui":
        return VAE_Zehui(hiddens=[16, 32, 64, 128, 256], latent_dim=128, seq_len=1000, out_channels=12)
    if model_name == "lstm_zehui":
        return LSTM_Zehui(in_channel=1, out_channel=12)
    if model_name == "ecgrecover":
        return ECGrecover_UMMISCO()
    if model_name == "mcma":
        return MCMA_UNet3Plus(input_ch=1, output_ch=12)
    if model_name == "unet1d_baseline":
        return UNet1D_Baseline(in_ch=1, out_ch=12, base_ch=32)
    if model_name == "diffusion_1to12":
        return Diffusion1to12(cond_ch=1, out_ch=12, base_ch=64, timesteps=200, sample_steps=30, val_sample_steps=20)
    if model_name == "beatdiff_1to12":
        return BeatDiff1to12(cond_ch=1, out_ch=12, base_ch=64, channel_mult=(1,2,3), num_blocks=2, sample_steps=25, val_sample_steps=20)

    if model_name == "cdgs":
        return CDGS(
            d_model=d_model or 256,
            n_gaussians=n_gaussians or 64,
            use_metadata=False,
            direct_lead_sum=args.cdgs_direct_lead_sum,
            direct_lead_patch=args.cdgs_direct_patch,
            direct_lead_span=args.cdgs_direct_span,
        )
    if model_name == "cdg_2":
        return CDGS2(
            d_model=d_model or 256,
            n_gaussians=n_gaussians or 128,
            use_metadata=False,
            direct_lead_sum=args.cdgs_direct_lead_sum,
            direct_lead_patch=args.cdgs_direct_patch,
            direct_lead_span=args.cdgs_direct_span,
        )
    if model_name == "cdg_3":
        return CDGS3(
            d_model=d_model or 256,
            n_gaussians=n_gaussians or 128,
            use_metadata=False,
            direct_lead_sum=args.cdgs_direct_lead_sum,
            direct_lead_patch=args.cdgs_direct_patch,
            direct_lead_span=args.cdgs_direct_span,
        )
    if model_name == "cdg_4":
        return CDGS4(
            d_model=d_model or 256,
            n_gaussians=n_gaussians or 64,
            use_metadata=False,
            direct_lead_sum=args.cdgs_direct_lead_sum,
            direct_lead_patch=args.cdgs_direct_patch,
            direct_lead_span=args.cdgs_direct_span,
        )
    if model_name == "cdg_5":
        return CDGS5(
            d_model=d_model or 256,
            n_gaussians=n_gaussians or 64,
            use_metadata=False,
            direct_lead_sum=args.cdgs_direct_lead_sum,
            direct_lead_patch=args.cdgs_direct_patch,
            direct_lead_span=args.cdgs_direct_span,
        )
    if model_name == "cdg_6":
        return CDGS6(
            d_model=d_model or 128,
            n_gaussians=n_gaussians or 2048,
            n_encoder_layers=n_encoder_layers or 2,
            use_metadata=False,
            direct_lead_sum=args.cdgs_direct_lead_sum,
            direct_lead_patch=args.cdgs_direct_patch,
            direct_lead_span=args.cdgs_direct_span,
            d_deform_hidden=d_deform_hidden or 128,
        )
    if model_name == "cdg_7":
        return CDGS7(
            d_model=d_model or 128,
            n_gaussians=n_gaussians or 512,
            n_encoder_layers=n_encoder_layers or 2,
            use_metadata=False,
            direct_lead_sum=args.cdgs_direct_lead_sum,
            direct_lead_patch=args.cdgs_direct_patch,
            direct_lead_span=args.cdgs_direct_span,
            d_deform_hidden=d_deform_hidden or 32,
        )
    if model_name == "cdg_8":
        return CDGS8(
            d_model=d_model or 128,
            n_gaussians=n_gaussians or 1024,
            n_encoder_layers=n_encoder_layers or 4,
            use_metadata=True,
        )
    if model_name == "cdg_9":
        return CDGS9(
            d_model=d_model or 256,
            n_gaussians=n_gaussians or 512,
            n_encoder_layers=n_encoder_layers or 4,
        )
    if model_name == "cdg_10":
        return CDGS10(
            d_model=d_model or 256,
            n_gaussians=n_gaussians or 512,
            n_encoder_layers=n_encoder_layers or 4,
        )
    if model_name == "cdg_11":
        return CDGS11(
            d_model=d_model or 256,
            n_gaussians=n_gaussians or 512,
            n_encoder_layers=n_encoder_layers or 4,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def checkpoint_weight_summary(model, state_dict):
    with torch.no_grad():
        total_params = sum(int(p.numel()) for p in model.parameters())
        trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)

        sq_sum = 0.0
        abs_sum = 0.0
        count = 0
        for _, p in model.named_parameters():
            data = p.detach().float()
            sq_sum += float(torch.sum(data * data).item())
            abs_sum += float(torch.sum(torch.abs(data)).item())
            count += int(data.numel())

        l2_norm = float(np.sqrt(sq_sum))
        mean_abs = float(abs_sum / max(count, 1))

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "weight_l2_norm": l2_norm,
        "weight_abs_mean": mean_abs,
        "state_dict_keys": len(state_dict),
    }


def forward_model(model_name, model, x, meta, input_idx):
    if model_name == "vae_zehui":
        pred, _, _ = model(x)
    elif model_name in ("cdgs", "cdg_2", "cdg_3", "cdg_4", "cdg_5", "cdg_6", "cdg_7"):
        age = meta[:, 0]
        sex = meta[:, 1].long()
        pred, _, _ = model(x, age, sex)
    elif model_name == "cdg_8":
        pred, _, _ = model(x, meta=meta, bypass_alpha=1.0)
    elif model_name in ("cdg_9", "cdg_10", "cdg_11"):
        lead_id = torch.full((x.size(0),), input_idx, dtype=torch.long, device=x.device)
        pred, _, _ = model(x, meta=meta, bypass_alpha=1.0, lead_id=lead_id)
    else:
        pred = model(x)
    return pred


def compute_metrics(pred_np, gt_np, input_idx, tolerance=0.10):
    n_leads = pred_np.shape[1]
    lead_indices = [i for i in range(n_leads) if i != input_idx]

    pred_wo = pred_np[:, lead_indices, :]
    gt_wo = gt_np[:, lead_indices, :]
    err = pred_wo - gt_wo

    mae_no_input = float(np.mean(np.abs(err)))
    rmse_no_input = float(np.sqrt(np.mean(err ** 2)))
    point_acc_no_input = float(np.mean(np.abs(err) <= tolerance))

    p_flat = pred_wo.reshape(-1)
    g_flat = gt_wo.reshape(-1)
    if np.std(p_flat) > 1e-8 and np.std(g_flat) > 1e-8:
        pearson_no_input = float(np.corrcoef(p_flat, g_flat)[0, 1])
    else:
        pearson_no_input = 0.0

    per_lead = {}
    for li in range(n_leads):
        p = pred_np[:, li, :].reshape(-1)
        g = gt_np[:, li, :].reshape(-1)
        lead_mae = float(np.mean(np.abs(p - g)))
        if np.std(p) > 1e-8 and np.std(g) > 1e-8:
            lead_pearson = float(np.corrcoef(p, g)[0, 1])
        else:
            lead_pearson = 0.0
        per_lead[LEADS[li]] = {
            "mae": lead_mae,
            "pearson": lead_pearson,
        }

    return {
        "mae_no_input": mae_no_input,
        "rmse_no_input": rmse_no_input,
        "point_accuracy_no_input": point_acc_no_input,
        "pearson_no_input": pearson_no_input,
        "acc_tolerance": float(tolerance),
        "per_lead": per_lead,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)
    file_col = "filename_hr" if args.use_hr else "filename_lr"
    meta_lookup = build_meta_lookup(df, file_col)

    test_folds_set = {int(x.strip()) for x in args.test_folds.split(",") if x.strip()}
    if "strat_fold" not in df.columns:
        raise ValueError("ptbxl_database.csv does not include strat_fold")

    test_df = df[df["strat_fold"].isin(test_folds_set)]
    test_records = test_df[file_col].tolist()
    if args.max_records > 0:
        test_records = test_records[:args.max_records]

    print(f"[*] TEST records: {len(test_records)} (strat_fold={args.test_folds})")

    input_idx = LEADS.index(args.input_lead)
    test_ds = PTBXLReconstructionDataset(data_dir, test_records, input_idx, meta_lookup)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[*] Loading checkpoint: {args.checkpoint}")
    ckpt_obj = torch.load(args.checkpoint, map_location=device)
    state_dict = extract_state_dict(ckpt_obj)
    arch_cfg = infer_arch_hparams(state_dict)

    model = build_model(args.model, args, arch_cfg).to(device)
    load_result = model.load_state_dict(state_dict, strict=args.strict_load)
    model.eval()

    weight_info = checkpoint_weight_summary(model, state_dict)

    ckpt_scalars = {}
    if isinstance(ckpt_obj, dict):
        for key in ("epoch", "best_val", "best_mae", "best_acc", "val_mae", "val_pearson"):
            if key in ckpt_obj and np.isscalar(ckpt_obj[key]):
                ckpt_scalars[key] = float(ckpt_obj[key])

    all_preds = []
    all_targets = []

    print("[*] Starting inference on TEST set...")
    with torch.no_grad():
        for x, y, meta in tqdm(test_loader, desc="inference"):
            x = x.to(device)
            y = y.to(device)
            meta = meta.to(device)

            pred = forward_model(args.model, model, x, meta, input_idx)
            if pred.size(-1) != y.size(-1):
                pred = F.interpolate(pred, size=y.size(-1), mode="linear", align_corners=False)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    pred_np = np.concatenate(all_preds, axis=0)
    gt_np = np.concatenate(all_targets, axis=0)

    pred_path = save_dir / "pred_12leads.npy"
    gt_path = save_dir / "gt_12leads.npy"
    np.save(pred_path, pred_np)
    np.save(gt_path, gt_np)

    metrics = compute_metrics(pred_np, gt_np, input_idx=input_idx, tolerance=args.acc_tolerance)
    summary = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "test_folds": sorted(list(test_folds_set)),
        "input_lead": args.input_lead,
        "num_samples": int(pred_np.shape[0]),
        "prediction_shape": list(pred_np.shape),
        "target_shape": list(gt_np.shape),
        "weights": weight_info,
        "checkpoint_scalars": ckpt_scalars,
        "load_strict": bool(args.strict_load),
        "missing_keys_count": len(load_result.missing_keys),
        "unexpected_keys_count": len(load_result.unexpected_keys),
        "missing_keys_preview": load_result.missing_keys[:20],
        "unexpected_keys_preview": load_result.unexpected_keys[:20],
        "metrics": metrics,
    }

    metrics_path = save_dir / "metrics_summary.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[*] Inference finished")
    print(f"[*] Pred shape: {pred_np.shape}")
    print(f"[*] GT shape:   {gt_np.shape}")
    print(f"[*] Saved pred: {pred_path}")
    print(f"[*] Saved gt:   {gt_path}")
    print(f"[*] Saved summary: {metrics_path}")

    print("\n[Checkpoint Weight Summary]")
    print(f"  total_params       : {weight_info['total_params']}")
    print(f"  trainable_params   : {weight_info['trainable_params']}")
    print(f"  weight_l2_norm     : {weight_info['weight_l2_norm']:.6f}")
    print(f"  weight_abs_mean    : {weight_info['weight_abs_mean']:.6f}")
    print(f"  missing_keys_count : {len(load_result.missing_keys)}")
    print(f"  unexpected_keys_count: {len(load_result.unexpected_keys)}")

    print("\n[Test Metrics]")
    print(f"  point_accuracy_no_input : {metrics['point_accuracy_no_input']:.6f}")
    print(f"  pearson_no_input        : {metrics['pearson_no_input']:.6f}")
    print(f"  mae_no_input            : {metrics['mae_no_input']:.6f}")
    print(f"  rmse_no_input           : {metrics['rmse_no_input']:.6f}")
    print(f"  acc_tolerance           : {metrics['acc_tolerance']}")


if __name__ == "__main__":
    main()
