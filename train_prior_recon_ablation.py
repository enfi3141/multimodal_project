from __future__ import print_function

import os
import ast
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from sklearn.metrics import f1_score, roc_auc_score

from models.resnet_lstm_ptbxl_raw import RawECGEncoder
from models.metadata_mlp import MetadataEncoder


LABEL_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def build_label_from_scp_codes(scp_codes_str, scp_statements):
    """
    PTB-XL scp_codes 문자열을 NORM, MI, STTC, CD, HYP multi-hot label로 변환.
    """
    y = np.zeros(len(LABEL_NAMES), dtype=np.float32)

    try:
        scp_dict = ast.literal_eval(scp_codes_str)
    except Exception:
        return y

    for code in scp_dict.keys():
        if code not in scp_statements.index:
            continue

        row = scp_statements.loc[code]

        if "diagnostic" in row and row["diagnostic"] != 1:
            continue

        cls = row.get("diagnostic_class", None)

        if cls in LABEL_NAMES:
            y[LABEL_NAMES.index(cls)] = 1.0

    return y


class PriorReconAblationDataset(data.Dataset):
    """
    reconstructions.npz 자체를 Dataset으로 사용.

    npz 내부:
    - inputs   : (N, 1, 1000)   현재 원본 1리드
    - preds    : (N, 12, 1000)  복원 12리드
    - targets  : (N, 12, 1000)  실제 현재 12리드
    - pairs    : (N, 3)         [past_path, current_path, delta_days]

    ptbxl_database.csv:
    - current_path == filename_lr 기준으로 label, age, sex 매칭
    """

    def __init__(
        self,
        npz_path,
        data_dir,
        use_time_delta=False,
    ):
        super().__init__()

        self.npz_path = npz_path
        self.data_dir = data_dir
        self.use_time_delta = use_time_delta

        z = np.load(npz_path)

        self.raw_1lead = z["inputs"].astype(np.float32)
        self.recon_12lead = z["preds"].astype(np.float32)
        self.real_12lead = z["targets"].astype(np.float32)
        self.past_12lead = z["past"].astype(np.float32) if "past" in z.files else None
        self.time_delta = z["time_delta"].astype(np.float32) if "time_delta" in z.files else None
        self.pairs = z["pairs"]

        def crop_to_length(x, target_len=2000):
            # x: (N, C, T)
            t = x.shape[-1]

            if t <= target_len:
                return x.astype(np.float32)

            start = (t - target_len) // 2
            end = start + target_len

            return x[:, :, start:end].astype(np.float32)


        if self.raw_1lead.shape[-1] > 2000:
            self.raw_1lead = crop_to_length(self.raw_1lead, target_len=2000)
            self.recon_12lead = crop_to_length(self.recon_12lead, target_len=2000)
            self.real_12lead = crop_to_length(self.real_12lead, target_len=2000)

            if self.past_12lead is not None:
                self.past_12lead = crop_to_length(self.past_12lead, target_len=2000)

            print("[INFO] Center-cropped signals to length:", self.raw_1lead.shape[-1])
        else:
            print("[INFO] Signal length:", self.raw_1lead.shape[-1])

        if self.raw_1lead.ndim != 3 or self.raw_1lead.shape[1] != 1:
            raise ValueError("inputs must have shape (N, 1, T), got {}".format(self.raw_1lead.shape))

        if self.recon_12lead.ndim != 3 or self.recon_12lead.shape[1] != 12:
            raise ValueError("preds must have shape (N, 12, T), got {}".format(self.recon_12lead.shape))

        if self.real_12lead.ndim != 3 or self.real_12lead.shape[1] != 12:
            raise ValueError("targets must have shape (N, 12, T), got {}".format(self.real_12lead.shape))

        self.current_paths = self.pairs[:, 1].astype(str)

        db_path = os.path.join(data_dir, "ptbxl_database.csv")
        scp_path = os.path.join(data_dir, "scp_statements.csv")

        df = pd.read_csv(db_path)
        scp_statements = pd.read_csv(scp_path, index_col=0)

        
        lr_set = set(df["filename_lr"].astype(str))
        hr_set = set(df["filename_hr"].astype(str))

        if self.current_paths[0] in lr_set:
            df = df.set_index("filename_lr")
            print("[INFO] Matched current paths with filename_lr")
        elif self.current_paths[0] in hr_set:
            df = df.set_index("filename_hr")
            print("[INFO] Matched current paths with filename_hr")
        else:
            raise ValueError(
                "Current paths are not matched with filename_lr or filename_hr. "
                "Example: {}".format(self.current_paths[0])
            )

        missing = [p for p in self.current_paths if p not in df.index]
        if len(missing) > 0:
            raise ValueError(
                "Some current paths are not found in ptbxl_database.csv. "
                "Example: {}".format(missing[:5])
            )

        labels = []
        metadata = []

        for p in self.current_paths:
            row = df.loc[p]

            label = build_label_from_scp_codes(row["scp_codes"], scp_statements)
            labels.append(label)

            age = safe_float(row["age"], default=60.0)
            age_norm = age / 100.0

            sex = safe_float(row["sex"], default=0.0)

            if sex == 1.0:
                sex_male = 1.0
                sex_female = 0.0
            else:
                sex_male = 0.0
                sex_female = 1.0

            if use_time_delta and self.time_delta is not None:
                td = float(self.time_delta[len(metadata), 0])
            else:
                td = 0.0

            meta = [
                age_norm,
                sex_male,
                sex_female,
                td,
            ]

            metadata.append(meta)

        self.labels = np.stack(labels).astype(np.float32)
        self.metadata = np.asarray(metadata, dtype=np.float32)

        print("[DATASET] {}".format(npz_path))
        print("  raw_1lead     :", self.raw_1lead.shape)
        print("  recon_12lead  :", self.recon_12lead.shape)
        print("  real_12lead   :", self.real_12lead.shape)
        print("  labels        :", self.labels.shape)
        print("  metadata      :", self.metadata.shape)
        print("  matched paths :", len(self.current_paths))

    def __len__(self):
        return len(self.raw_1lead)

    def __getitem__(self, idx):
        return {
            "raw_1lead": torch.tensor(self.raw_1lead[idx], dtype=torch.float32),
            "recon_12lead": torch.tensor(self.recon_12lead[idx], dtype=torch.float32),
            "real_12lead": torch.tensor(self.real_12lead[idx], dtype=torch.float32),
            "metadata": torch.tensor(self.metadata[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "current_path": self.current_paths[idx],
        }


class SingleECGClassifier(nn.Module):
    def __init__(
        self,
        input_key,
        in_channels,
        num_classes=5,
        feature_dim=128,
        lstm_hidden=128,
        lstm_layers=1,
        layers=(2, 2, 2),
        base_channels=32,
        dropout=0.2,
        use_meta=False,
        meta_in_dim=3,
        meta_feature_dim=16,
        fusion_type="concat",
    ):
        super().__init__()

        self.input_key = input_key
        self.use_meta = use_meta
        self.fusion_type = fusion_type

        self.ecg_encoder = RawECGEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            layers=layers,
            base_channels=base_channels,
        )

        if use_meta:
            self.meta_encoder = MetadataEncoder(
                in_dim=meta_in_dim,
                hidden_dim=32,
                feature_dim=meta_feature_dim,
                dropout=0.1,
            )
        else:
            self.meta_encoder = None

        if not use_meta:
            fusion_dim = feature_dim

        elif fusion_type == "concat":
            fusion_dim = feature_dim + meta_feature_dim

        elif fusion_type == "weighted":
            self.meta_proj = nn.Linear(meta_feature_dim, feature_dim)
            self.branch_weights = nn.Parameter(torch.ones(2))
            fusion_dim = feature_dim

        elif fusion_type == "gated":
            self.meta_proj = nn.Linear(meta_feature_dim, feature_dim)
            self.gate = nn.Sequential(
                nn.Linear(feature_dim + feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid(),
            )
            fusion_dim = feature_dim

        else:
            raise ValueError("Unknown fusion_type: {}".format(fusion_type))

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, batch):
        x = batch[self.input_key]
        ecg_feat = self.ecg_encoder(x)

        if not self.use_meta:
            fused = ecg_feat

        else:
            meta_feat = self.meta_encoder(batch["metadata"])

            if self.fusion_type == "concat":
                fused = torch.cat([ecg_feat, meta_feat], dim=1)

            elif self.fusion_type == "weighted":
                meta_feat = self.meta_proj(meta_feat)
                weights = torch.softmax(self.branch_weights, dim=0)
                fused = weights[0] * ecg_feat + weights[1] * meta_feat

            elif self.fusion_type == "gated":
                meta_feat = self.meta_proj(meta_feat)
                gate_input = torch.cat([ecg_feat, meta_feat], dim=1)
                gate = self.gate(gate_input)
                fused = gate * ecg_feat + (1.0 - gate) * meta_feat

            else:
                raise ValueError("Unknown fusion_type: {}".format(self.fusion_type))

        return self.classifier(fused)


class RawReconClassifier(nn.Module):
    def __init__(
        self,
        num_classes=5,
        raw_feature_dim=128,
        recon_feature_dim=128,
        lstm_hidden=128,
        lstm_layers=1,
        layers=(2, 2, 2),
        base_channels=32,
        dropout=0.2,
        use_meta=False,
        meta_in_dim=3,
        meta_feature_dim=16,
        fusion_type="concat",
    ):
        super().__init__()

        self.use_meta = use_meta
        self.fusion_type = fusion_type

        self.raw_encoder = RawECGEncoder(
            in_channels=1,
            feature_dim=raw_feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            layers=layers,
            base_channels=base_channels,
        )

        self.recon_encoder = RawECGEncoder(
            in_channels=12,
            feature_dim=recon_feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            layers=layers,
            base_channels=base_channels,
        )

        if use_meta:
            self.meta_encoder = MetadataEncoder(
                in_dim=meta_in_dim,
                hidden_dim=32,
                feature_dim=meta_feature_dim,
                dropout=0.1,
            )
        else:
            self.meta_encoder = None

        # concat fusion
        if fusion_type == "concat":
            fusion_dim = raw_feature_dim + recon_feature_dim
            if use_meta:
                fusion_dim += meta_feature_dim

            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        # weighted fusion: raw/recon을 같은 128차원에서 가중합
        elif fusion_type == "weighted":
            if raw_feature_dim != recon_feature_dim:
                raise ValueError("weighted fusion requires raw_feature_dim == recon_feature_dim")

            self.branch_weights = nn.Parameter(torch.ones(2))

            fusion_dim = raw_feature_dim
            if use_meta:
                fusion_dim += meta_feature_dim

            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        # gated fusion: raw/recon feature를 보고 sample별 gate 생성
        elif fusion_type == "gated":
            if raw_feature_dim != recon_feature_dim:
                raise ValueError("gated fusion requires raw_feature_dim == recon_feature_dim")

            self.gate = nn.Sequential(
                nn.Linear(raw_feature_dim + recon_feature_dim, raw_feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(raw_feature_dim, raw_feature_dim),
                nn.Sigmoid(),
            )

            fusion_dim = raw_feature_dim
            if use_meta:
                fusion_dim += meta_feature_dim

            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        else:
            raise ValueError("Unknown fusion_type: {}".format(fusion_type))

    def forward(self, batch):
        raw_feat = self.raw_encoder(batch["raw_1lead"])
        recon_feat = self.recon_encoder(batch["recon_12lead"])

        meta_feat = None
        if self.use_meta:
            meta_feat = self.meta_encoder(batch["metadata"])

        if self.fusion_type == "concat":
            feats = [raw_feat, recon_feat]
            if self.use_meta:
                feats.append(meta_feat)
            fused = torch.cat(feats, dim=1)

        elif self.fusion_type == "weighted":
            weights = torch.softmax(self.branch_weights, dim=0)
            fused_ecg = weights[0] * raw_feat + weights[1] * recon_feat

            if self.use_meta:
                fused = torch.cat([fused_ecg, meta_feat], dim=1)
            else:
                fused = fused_ecg

        elif self.fusion_type == "gated":
            gate_input = torch.cat([raw_feat, recon_feat], dim=1)
            gate = self.gate(gate_input)

            fused_ecg = gate * raw_feat + (1.0 - gate) * recon_feat

            if self.use_meta:
                fused = torch.cat([fused_ecg, meta_feat], dim=1)
            else:
                fused = fused_ecg

        else:
            raise ValueError("Unknown fusion_type: {}".format(self.fusion_type))

        return self.classifier(fused)


def build_model(args, meta_in_dim):
    if args.experiment == "raw1":
        return SingleECGClassifier(
            input_key="raw_1lead",
            in_channels=1,
            num_classes=args.num_classes,
            use_meta=False,
            meta_in_dim=meta_in_dim,
        )
    
    elif args.experiment == "raw1_meta":
        return SingleECGClassifier(
            input_key="raw_1lead",
            in_channels=1,
            num_classes=args.num_classes,
            use_meta=True,
            meta_in_dim=meta_in_dim,
            fusion_type=args.fusion,
        )

    elif args.experiment == "recon12":
        return SingleECGClassifier(
            input_key="recon_12lead",
            in_channels=12,
            num_classes=args.num_classes,
            use_meta=False,
            meta_in_dim=meta_in_dim,
        )
    
    elif args.experiment == "recon12_meta":
        return SingleECGClassifier(
            input_key="recon_12lead",
            in_channels=12,
            num_classes=args.num_classes,
            use_meta=True,
            meta_in_dim=meta_in_dim,
            fusion_type=args.fusion,
        )

    elif args.experiment == "raw1_recon12":
        return RawReconClassifier(
            num_classes=args.num_classes,
            use_meta=False,
            meta_in_dim=meta_in_dim,
            fusion_type=args.fusion,
        )

    elif args.experiment == "raw1_recon12_meta":
        return RawReconClassifier(
            num_classes=args.num_classes,
            use_meta=True,
            meta_in_dim=meta_in_dim,
            fusion_type=args.fusion,
        )

    elif args.experiment == "real12":
        return SingleECGClassifier(
            input_key="real_12lead",
            in_channels=12,
            num_classes=args.num_classes,
            use_meta=False,
            meta_in_dim=meta_in_dim,
        )

    elif args.experiment == "real12_meta":
        return SingleECGClassifier(
            input_key="real_12lead",
            in_channels=12,
            num_classes=args.num_classes,
            use_meta=True,
            meta_in_dim=meta_in_dim,
            fusion_type=args.fusion,
        )

    else:
        raise ValueError("Unknown experiment: {}".format(args.experiment))


def build_loader(args, split):
    if split == "train":
        npz_path = args.train_npz
    elif split == "val":
        npz_path = args.val_npz
    elif split == "test":
        npz_path = args.test_npz
    else:
        raise ValueError("Unknown split: {}".format(split))

    dataset = PriorReconAblationDataset(
        npz_path=npz_path,
        data_dir=args.data,
        use_time_delta=args.use_time_delta,
    )

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.workers,
        pin_memory=True,
    )

    return loader, dataset


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Multi-label classification metrics.

    Returns:
      macro_f1
      macro_auc
      accuracy
      sensitivity
      specificity
      per_label_auc
      per_label_detail
    """

    y_pred = (y_prob >= threshold).astype(np.float32)

    # 1) Accuracy: 전체 label position 기준 정확도
    # shape: (N, 5)이므로 NORM, MI, STTC, CD, HYP 전체에 대해 평균
    accuracy = float((y_pred == y_true).mean())

    # 2) Macro F1
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    per_label_auc = {}
    per_label_detail = {}

    auc_values = []
    sensitivity_values = []
    specificity_values = []

    for i, name in enumerate(LABEL_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        yscore = y_prob[:, i]

        # AUROC
        try:
            auc = roc_auc_score(yt, yscore)
        except ValueError:
            auc = float("nan")

        per_label_auc[name] = float(auc)

        if not np.isnan(auc):
            auc_values.append(auc)

        # Confusion matrix components
        tp = np.sum((yt == 1) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))

        # Sensitivity = Recall = TP / (TP + FN)
        if tp + fn > 0:
            sensitivity = tp / (tp + fn)
            sensitivity_values.append(sensitivity)
        else:
            sensitivity = float("nan")

        # Specificity = TN / (TN + FP)
        if tn + fp > 0:
            specificity = tn / (tn + fp)
            specificity_values.append(specificity)
        else:
            specificity = float("nan")

        per_label_detail[name] = {
            "auc": float(auc),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    macro_auc = float(np.mean(auc_values)) if len(auc_values) > 0 else float("nan")
    macro_sensitivity = float(np.mean(sensitivity_values)) if len(sensitivity_values) > 0 else float("nan")
    macro_specificity = float(np.mean(specificity_values)) if len(specificity_values) > 0 else float("nan")

    return (
        float(macro_f1),
        macro_auc,
        accuracy,
        macro_sensitivity,
        macro_specificity,
        per_label_auc,
        per_label_detail,
    )

def find_best_thresholds(y_true, y_prob):
    """
    Find class-wise thresholds on validation set by maximizing per-class F1.
    These thresholds should be selected only on validation data,
    then applied to test data.
    """
    thresholds = []
    grid = np.arange(0.05, 0.51, 0.01)

    for i in range(y_true.shape[1]):
        best_th = 0.5
        best_f1 = -1.0

        for th in grid:
            y_pred_i = (y_prob[:, i] >= th).astype(np.float32)
            f1 = f1_score(
                y_true[:, i],
                y_pred_i,
                average="binary",
                zero_division=0,
            )

            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        thresholds.append(best_th)

    return np.asarray(thresholds, dtype=np.float32)


@torch.no_grad()
def predict_probs(loader, model, device):
    model.eval()

    all_targets = []
    all_probs = []

    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)

        targets = batch["label"].float()
        outputs = model(batch)
        probs = torch.sigmoid(outputs)

        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    return y_true, y_prob


def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)

        targets = batch["label"].float()
        outputs = model(batch)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0

    all_targets = []
    all_probs = []

    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)

        targets = batch["label"].float()
        outputs = model(batch)

        loss = criterion(outputs, targets)
        probs = torch.sigmoid(outputs)

        total_loss += loss.item() * targets.size(0)

        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    (
        macro_f1,
        macro_auc,
        accuracy,
        sensitivity,
        specificity,
        per_label_auc,
        per_label_detail,
    ) = compute_metrics(y_true, y_prob)

    avg_loss = total_loss / len(loader.dataset)

    return (
        avg_loss,
        macro_f1,
        macro_auc,
        accuracy,
        sensitivity,
        specificity,
        per_label_auc,
        per_label_detail,
    )


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="PTB-XL prior reconstruction ablation training"
    )

    parser.add_argument(
        "--fusion",
        type=str,
        default="concat",
        choices=["concat", "weighted", "gated"],
    )

    parser.add_argument("--data", type=str, required=True)

    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--test_npz", type=str, required=True)

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[
            "raw1",
            "raw1_meta",
            "recon12",
            "recon12_meta",
            "raw1_recon12",
            "raw1_recon12_meta",
            "real12",
            "real12_meta",
        ],
    )

    parser.add_argument("--use_time_delta", action="store_true")

    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_ptbxl/prior_ablation")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(args.seed)

    save_name = args.experiment

    if args.experiment in ["raw1_meta", "recon12_meta", "raw1_recon12", "raw1_recon12_meta", "real12_meta"]:
        save_name = "{}_{}".format(args.experiment, args.fusion)

    save_dir = os.path.join(args.checkpoint, save_name)
    os.makedirs(save_dir, exist_ok=True)

    print("[INFO] Experiment:", args.experiment)
    print("[INFO] Device:", device)
    print("[INFO] Save dir:", save_dir)

    trainloader, train_ds = build_loader(args, "train")
    valloader, val_ds = build_loader(args, "val")
    testloader, test_ds = build_loader(args, "test")

    meta_in_dim = train_ds.metadata.shape[1]

    model = build_model(args, meta_in_dim=meta_in_dim)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_auc = -1.0
    best_epoch = -1
    history = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            trainloader, model, criterion, optimizer, device
        )

        (
            val_loss,
            val_f1,
            val_auc,
            val_acc,
            val_sens,
            val_spec,
            val_per_label_auc,
            val_per_label_detail,
        ) = evaluate(valloader, model, criterion, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_f1),
            "val_macro_auc": float(val_auc),
            "val_sensitivity": float(val_sens),
            "val_specificity": float(val_spec),
            "val_per_label_auc": val_per_label_auc,
            "val_per_label_detail": val_per_label_detail,
        }
        history.append(row)

        print(
            "[Epoch {}/{}] train_loss={:.4f} val_loss={:.4f} "
            "val_acc={:.4f} val_macro_f1={:.4f} val_macro_auc={:.4f} "
            "val_sens={:.4f} val_spec={:.4f}".format(
                epoch + 1,
                args.epochs,
                train_loss,
                val_loss,
                val_acc,
                val_f1,
                val_auc,
                val_sens,
                val_spec,
            )
        )

        print(
            "  " + " | ".join(
                ["{}={:.4f}".format(k, v) for k, v in val_per_label_auc.items()]
            )
        )

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1

            ckpt_path = os.path.join(save_dir, "best_model.pth")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "args": vars(args),
                },
                ckpt_path,
            )

            print("[SAVE]", ckpt_path)

        save_json(os.path.join(save_dir, "history.json"), history)

    print("[BEST] epoch={}, val_macro_auc={:.4f}".format(best_epoch, best_auc))

    ckpt_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # 1) Find class-wise thresholds on validation set
    val_y_true, val_y_prob = predict_probs(valloader, model, device)
    best_thresholds = find_best_thresholds(val_y_true, val_y_prob)

    print(
        "[BEST THRESHOLDS] "
        + " | ".join(
            ["{}={:.2f}".format(name, th) for name, th in zip(LABEL_NAMES, best_thresholds)]
        )
    )

    # 2) Keep loss from normal evaluation if needed
    (
        test_loss,
        _test_f1_05,
        _test_auc_05,
        _test_acc_05,
        _test_sens_05,
        _test_spec_05,
        _test_per_label_auc_05,
        _test_per_label_detail_05,
    ) = evaluate(testloader, model, criterion, device)

    # 3) Recompute test metrics using validation-selected thresholds
    test_y_true, test_y_prob = predict_probs(testloader, model, device)

    (
        test_f1,
        test_auc,
        test_acc,
        test_sens,
        test_spec,
        test_per_label_auc,
        test_per_label_detail,
    ) = compute_metrics(
        test_y_true,
        test_y_prob,
        threshold=best_thresholds,
    )

    test_metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_macro_auc": float(test_auc),
        "test_sensitivity": float(test_sens),
        "test_specificity": float(test_spec),
        "test_per_label_auc": test_per_label_auc,
        "test_per_label_detail": test_per_label_detail,
        "best_thresholds": {
            name: float(th)
            for name, th in zip(LABEL_NAMES, best_thresholds)
        },
        "best_epoch": int(best_epoch),
        "best_val_macro_auc": float(best_auc),
        "label_names": LABEL_NAMES,
    }

    save_json(os.path.join(save_dir, "test_metrics.json"), test_metrics)

    print(
        "[TEST] loss={:.4f} acc={:.4f} macro_f1={:.4f} macro_auc={:.4f} "
        "sens={:.4f} spec={:.4f}".format(
            test_loss,
            test_acc,
            test_f1,
            test_auc,
            test_sens,
            test_spec,
        )
    )
    print(
        "  " + " | ".join(
            ["{}={:.4f}".format(k, v) for k, v in test_per_label_auc.items()]
        )
    )


if __name__ == "__main__":
    main()