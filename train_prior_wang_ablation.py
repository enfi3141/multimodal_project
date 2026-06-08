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
from models.resnet1d_wang import ResNet1dWang


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
    Same label mapping logic as the Wang direct evaluation code.
    - ignore scp code with score <= 0
    - map diagnostic superclass through diagnostic_class
    - samples with no matched labels will be removed later
    """
    y = np.zeros(len(LABEL_NAMES), dtype=np.float32)

    try:
        scp_dict = ast.literal_eval(scp_codes_str) if isinstance(scp_codes_str, str) else scp_codes_str
    except Exception:
        return y

    for code, score in scp_dict.items():
        try:
            if float(score) <= 0:
                continue
        except Exception:
            continue

        code = str(code)

        if code in LABEL_NAMES:
            cls = code
        else:
            if code not in scp_statements.index:
                continue

            row = scp_statements.loc[code]

            if "diagnostic" in row and not bool(row["diagnostic"]):
                continue

            cls = str(row.get("diagnostic_class", ""))

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
        rebuild_real_from_wfdb=False,
    ):
        super().__init__()

        self.npz_path = npz_path
        self.data_dir = data_dir
        self.use_time_delta = use_time_delta
        self.rebuild_real_from_wfdb = rebuild_real_from_wfdb

        z = np.load(npz_path, allow_pickle=True)

        self.raw_1lead = z["inputs"].astype(np.float32)
        self.recon_12lead = z["preds"].astype(np.float32)
        self.real_12lead = z["targets"].astype(np.float32)
        self.past_12lead = z["past"].astype(np.float32) if "past" in z.files else None
        self.time_delta = z["time_delta"].astype(np.float32) if "time_delta" in z.files else None
        self.pairs = z["pairs"]

        def first_crop_to_1000(x):
            # x: (N, C, T)
            crop_len = 1000
            t = x.shape[-1]

            if t > crop_len:
                return x[..., :crop_len].astype(np.float32)

            if t == crop_len:
                return x.astype(np.float32)

            raise ValueError("Signal length is shorter than 1000: got {}".format(t))


        def zscore_per_lead(x):
            # x: (N, C, T)
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True)
            return ((x - mean) / (std + 1e-8)).astype(np.float32)


        self.raw_1lead = first_crop_to_1000(self.raw_1lead)
        self.recon_12lead = first_crop_to_1000(self.recon_12lead)
        self.real_12lead = first_crop_to_1000(self.real_12lead)

        if self.past_12lead is not None:
            self.past_12lead = first_crop_to_1000(self.past_12lead)

        print("[INFO] First-cropped signals to length:", self.raw_1lead.shape[-1])

        self.raw_1lead = zscore_per_lead(self.raw_1lead)
        self.recon_12lead = zscore_per_lead(self.recon_12lead)
        self.real_12lead = zscore_per_lead(self.real_12lead)

        if self.past_12lead is not None:
            self.past_12lead = zscore_per_lead(self.past_12lead)

        print("[INFO] Applied per-lead z-score normalization.")

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
        
        def load_wfdb_12lead_first1000(rel_path):
            import wfdb

            full_path = os.path.join(data_dir, rel_path)
            sig = wfdb.rdrecord(full_path).p_signal.astype(np.float32).T  # (12, T)
            sig = sig[:, :1000]
            sig = (sig - sig.mean(axis=1, keepdims=True)) / (
                sig.std(axis=1, keepdims=True) + 1e-8
            )
            return sig.astype(np.float32)

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

            meta = [
                age_norm,
                sex_male,
                sex_female,
            ]

            metadata.append(meta)

        self.labels = np.stack(labels).astype(np.float32)
        self.metadata = np.asarray(metadata, dtype=np.float32)

        valid_mask = self.labels.sum(axis=1) > 0

        if not np.all(valid_mask):
            before = len(valid_mask)

            self.raw_1lead = self.raw_1lead[valid_mask]
            self.recon_12lead = self.recon_12lead[valid_mask]
            self.real_12lead = self.real_12lead[valid_mask]

            if self.past_12lead is not None:
                self.past_12lead = self.past_12lead[valid_mask]

            if self.time_delta is not None:
                self.time_delta = self.time_delta[valid_mask]

            self.pairs = self.pairs[valid_mask]
            self.current_paths = self.current_paths[valid_mask]
            self.labels = self.labels[valid_mask]
            self.metadata = self.metadata[valid_mask]

            after = len(self.current_paths)
            print("[INFO] Removed samples without diagnostic labels: {} -> {}".format(before, after))

        if self.rebuild_real_from_wfdb and len(self.current_paths) > 0 and self.current_paths[0] in hr_set:
            self.real_12lead = np.stack(
                [load_wfdb_12lead_first1000(p) for p in self.current_paths],
                axis=0,
            ).astype(np.float32)

            print("[INFO] Rebuilt real_12lead from WFDB filename_hr using Wang preprocessing.")

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


class WangECGEncoder(nn.Module):
    def __init__(self, n_leads=12):
        super().__init__()
        self.backbone = ResNet1dWang(n_leads=n_leads, n_classes=5)

    def forward(self, x):
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)

        x = self.backbone.head[0](x)  # AdaptiveConcatPool1d: (B, 256)
        x = self.backbone.head[1](x)  # BN
        x = self.backbone.head[2](x)  # Dropout
        x = self.backbone.head[3](x)  # Linear 256 -> 128
        x = self.backbone.head[4](x)  # ReLU

        return x  # (B, 128)

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

        self.ecg_encoder = WangECGEncoder(
            n_leads=in_channels
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

class WangLogitMetaClassifier(nn.Module):
    """
    Full pretrained Wang classifier + metadata fusion.

    ECG branch:
      ResNet1dWang(x) -> wang_logits (B, 5)

    Metadata branch:
      MetadataEncoder(meta) -> meta_feat (B, 16)
      meta_to_logits(meta_feat) -> meta_logits (B, 5)

    Fusion:
      concat / weighted / gated
    """

    def __init__(
        self,
        input_key,
        in_channels=12,
        num_classes=5,
        meta_in_dim=3,
        meta_feature_dim=16,
        fusion_type="concat",
        dropout=0.2,
    ):
        super().__init__()

        self.input_key = input_key
        self.fusion_type = fusion_type
        self.num_classes = num_classes

        # Full Wang classifier: output is already disease logits (B, 5)
        self.wang = ResNet1dWang(n_leads=in_channels, n_classes=num_classes)

        self.meta_encoder = MetadataEncoder(
            in_dim=meta_in_dim,
            hidden_dim=32,
            feature_dim=meta_feature_dim,
            dropout=0.1,
        )

        if fusion_type == "concat":
            # wang_logits(5) + meta_feat(16) -> final logits(5)
            fusion_dim = num_classes + meta_feature_dim
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(32, num_classes),
            )

        elif fusion_type == "weighted":
            # meta feature -> metadata logits(5), then weighted sum with Wang logits
            self.meta_to_logits = nn.Linear(meta_feature_dim, num_classes)
            self.branch_weights = nn.Parameter(torch.ones(2))

        elif fusion_type == "gated":
            # sample-wise, class-wise gate between Wang logits and metadata logits
            self.meta_to_logits = nn.Linear(meta_feature_dim, num_classes)
            self.gate = nn.Sequential(
                nn.Linear(num_classes + num_classes, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, num_classes),
                nn.Sigmoid(),
            )

        else:
            raise ValueError("Unknown fusion_type: {}".format(fusion_type))

    def forward(self, batch):
        x = batch[self.input_key]

        wang_logits = self.wang(x)  # (B, 5)
        meta_feat = self.meta_encoder(batch["metadata"])  # (B, 16)

        if self.fusion_type == "concat":
            fused = torch.cat([wang_logits, meta_feat], dim=1)
            return self.classifier(fused)

        elif self.fusion_type == "weighted":
            meta_logits = self.meta_to_logits(meta_feat)  # (B, 5)
            weights = torch.softmax(self.branch_weights, dim=0)
            return weights[0] * wang_logits + weights[1] * meta_logits

        elif self.fusion_type == "gated":
            meta_logits = self.meta_to_logits(meta_feat)  # (B, 5)
            gate_input = torch.cat([wang_logits, meta_logits], dim=1)
            gate = self.gate(gate_input)  # (B, 5)
            return gate * wang_logits + (1.0 - gate) * meta_logits

        else:
            raise ValueError("Unknown fusion_type: {}".format(self.fusion_type))

class WangFeatureGatedMetaClassifier(nn.Module):
    """
    Feature-level gated fusion using pretrained Wang feature space.

    ECG branch:
      ResNet1dWang backbone + head[0:5] -> h_ecg (B, 128)

    Metadata branch:
      MetadataEncoder -> meta_feat (B, 16)
      Linear(16 -> 128) -> h_meta (B, 128)

    Gate:
      g = sigmoid(MLP([h_ecg; h_meta]))  # (B, 128)
      h_fused = g * h_ecg + (1 - g) * h_meta

    Classifier:
      Wang head[5:] -> logits (B, 5)
    """

    def __init__(
        self,
        input_key,
        in_channels=12,
        num_classes=5,
        meta_in_dim=3,
        meta_feature_dim=16,
        feature_dim=128,
    ):
        super().__init__()

        self.input_key = input_key
        self.feature_dim = feature_dim

        base = ResNet1dWang(n_leads=in_channels, n_classes=num_classes)

        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        head_layers = list(base.head.children())

        # [0] AdaptiveConcatPool, [1] BN(256), [2] Dropout,
        # [3] Linear(256->128), [4] ReLU
        self.head_feat = nn.Sequential(*head_layers[:5])

        # [5] BN(128), [6] Dropout, [7] Linear(128->5)
        self.head_cls = nn.Sequential(*head_layers[5:])

        self.meta_encoder = MetadataEncoder(
            in_dim=meta_in_dim,
            hidden_dim=32,
            feature_dim=meta_feature_dim,
            dropout=0.1,
        )

        self.meta_proj = nn.Linear(meta_feature_dim, feature_dim)

        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

    def extract_ecg_feat(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.head_feat(x)
        return x

    def forward(self, batch):
        x = batch[self.input_key]

        h_ecg = self.extract_ecg_feat(x)                  # (B, 128)
        meta_feat = self.meta_encoder(batch["metadata"])  # (B, 16)
        h_meta = self.meta_proj(meta_feat)                # (B, 128)

        gate_input = torch.cat([h_ecg, h_meta], dim=1)
        gate = self.gate(gate_input)                      # (B, 128)

        h_fused = gate * h_ecg + (1.0 - gate) * h_meta
        logits = self.head_cls(h_fused)

        return logits

class WangDirectClassifier(nn.Module):
    """
    Direct use of the full pretrained Wang classifier.

    ECG branch:
      ResNet1dWang(x) -> logits (B, 5)

    Used for recon12 / real12 when reproducing the pretrained
    Wang classifier output without adding a new classification head.
    """

    def __init__(
        self,
        input_key,
        in_channels=12,
        num_classes=5,
    ):
        super().__init__()
        self.input_key = input_key
        self.wang = ResNet1dWang(n_leads=in_channels, n_classes=num_classes)

    def forward(self, batch):
        x = batch[self.input_key]
        return self.wang(x)

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
    
    elif args.experiment == "recon12_meta_feature_gate":
        return WangFeatureGatedMetaClassifier(
            input_key="recon_12lead",
            in_channels=12,
            num_classes=args.num_classes,
            meta_in_dim=meta_in_dim,
        )

    elif args.experiment == "real12_meta_feature_gate":
        return WangFeatureGatedMetaClassifier(
            input_key="real_12lead",
            in_channels=12,
            num_classes=args.num_classes,
            meta_in_dim=meta_in_dim,
        )
    
    elif args.experiment == "recon12_meta_logit":
        return WangLogitMetaClassifier(
            input_key="recon_12lead",
            in_channels=12,
            num_classes=args.num_classes,
            meta_in_dim=meta_in_dim,
            fusion_type=args.fusion,
        )

    elif args.experiment == "real12_meta_logit":
        return WangLogitMetaClassifier(
            input_key="real_12lead",
            in_channels=12,
            num_classes=args.num_classes,
            meta_in_dim=meta_in_dim,
            fusion_type=args.fusion,
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
        return WangDirectClassifier(
            input_key="recon_12lead",
            in_channels=12,
            num_classes=args.num_classes,
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
        return WangDirectClassifier(
            input_key="real_12lead",
            in_channels=12,
            num_classes=args.num_classes,
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
    

def strip_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    return new_state


def load_pretrained_wang(model, ckpt_path, device):
    print("[INFO] Loading pretrained Wang checkpoint:", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    state = strip_module_prefix(state)

    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    if hasattr(target_model, "head_feat") and hasattr(target_model, "head_cls"):
        print("[INFO] Loading into feature-level Wang gated model")

        mapped_state = {}

        for k, v in state.items():
            # backbone keys are same
            if k.startswith(("stem.", "layer1.", "layer2.", "layer3.")):
                mapped_state[k] = v

            # original Wang head -> split head
            elif k.startswith("head.0."):
                mapped_state[k.replace("head.0.", "head_feat.0.")] = v
            elif k.startswith("head.1."):
                mapped_state[k.replace("head.1.", "head_feat.1.")] = v
            elif k.startswith("head.2."):
                mapped_state[k.replace("head.2.", "head_feat.2.")] = v
            elif k.startswith("head.3."):
                mapped_state[k.replace("head.3.", "head_feat.3.")] = v
            elif k.startswith("head.4."):
                mapped_state[k.replace("head.4.", "head_feat.4.")] = v
            elif k.startswith("head.5."):
                mapped_state[k.replace("head.5.", "head_cls.0.")] = v
            elif k.startswith("head.6."):
                mapped_state[k.replace("head.6.", "head_cls.1.")] = v
            elif k.startswith("head.7."):
                mapped_state[k.replace("head.7.", "head_cls.2.")] = v

        missing, unexpected = target_model.load_state_dict(
            mapped_state,
            strict=False,
        )

        print("[INFO] Loaded pretrained Wang checkpoint into feature-gated model.")
        print("[INFO] Missing keys:", missing)
        print("[INFO] Unexpected keys:", unexpected)
        return

    if hasattr(target_model, "ecg_encoder") and hasattr(target_model.ecg_encoder, "backbone"):
        load_target = target_model.ecg_encoder.backbone
        print("[INFO] Loading into ecg_encoder.backbone")

    elif hasattr(target_model, "wang"):
        load_target = target_model.wang
        print("[INFO] Loading into full Wang classifier")

    else:
        raise ValueError("No compatible Wang model found for pretrained loading.")

    missing, unexpected = load_target.load_state_dict(
        state,
        strict=False,
    )

    print("[INFO] Loaded pretrained Wang checkpoint.")
    print("[INFO] Missing keys:", missing)
    print("[INFO] Unexpected keys:", unexpected)


def build_loader(args, split):
    if split == "train":
        npz_path = args.train_npz
    elif split == "val":
        npz_path = args.val_npz
    elif split == "test":
        npz_path = args.test_npz
    else:
        raise ValueError("Unknown split: {}".format(split))

    rebuild_real_from_wfdb = args.experiment in [
        "real12",
        "real12_meta",
        "real12_meta_logit",
        "real12_meta_feature_gate",
    ]

    dataset = PriorReconAblationDataset(
        npz_path=npz_path,
        data_dir=args.data,
        use_time_delta=args.use_time_delta,
        rebuild_real_from_wfdb=rebuild_real_from_wfdb,
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

    threshold = np.asarray(threshold, dtype=np.float32)
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
def collect_probs(loader, model, criterion, device):
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
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, y_true, y_prob


def find_optimal_thresholds(y_true, y_prob, metric="f1", grid=None):
    """
    Validation set에서 class별 threshold를 찾는다.
    - metric='f1': label별 F1 최대화
    - metric='youden': sensitivity + specificity - 1 최대화
    """
    if grid is None:
        grid = np.arange(0.05, 0.951, 0.01)

    thresholds = []
    details = {}

    for i, name in enumerate(LABEL_NAMES):
        yt = y_true[:, i]
        scores = y_prob[:, i]

        best_t = 0.5
        best_score = -1.0
        best_sens = 0.0
        best_spec = 0.0

        for t in grid:
            yp = (scores >= t).astype(np.float32)

            tp = np.sum((yt == 1) & (yp == 1))
            tn = np.sum((yt == 0) & (yp == 0))
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            if metric == "f1":
                denom = 2 * tp + fp + fn
                score = (2 * tp / denom) if denom > 0 else 0.0
            elif metric == "youden":
                score = sens + spec - 1.0
            else:
                raise ValueError("Unknown threshold metric: {}".format(metric))

            if score > best_score:
                best_score = score
                best_t = float(t)
                best_sens = float(sens)
                best_spec = float(spec)

        thresholds.append(best_t)
        details[name] = {
            "threshold": float(best_t),
            "best_{}_score".format(metric): float(best_score),
            "sensitivity_at_threshold": float(best_sens),
            "specificity_at_threshold": float(best_spec),
        }

    return np.asarray(thresholds, dtype=np.float32), details


@torch.no_grad()
def evaluate(loader, model, criterion, device, threshold=0.5):
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
    ) = compute_metrics(y_true, y_prob, threshold=threshold)

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
    parser.add_argument(
        "--pretrained_wang",
        type=str,
        default=None,
        help="Path to pretrained ResNet1dWang checkpoint, e.g., resnet1d_wang_best.pt",
    )

    parser.add_argument(
        "--freeze_wang",
        action="store_true",
        help="Freeze pretrained Wang ECG encoder after loading.",
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
            "recon12_meta_logit",
            "real12_meta_logit",
            "recon12_meta_feature_gate",
            "real12_meta_feature_gate",
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
    parser.add_argument(
        "--threshold_metric",
        type=str,
        default="f1",
        choices=["f1", "youden"],
        help="Metric used to optimize class-wise thresholds on validation set.",
    )
    parser.add_argument(
        "--optimize_threshold",
        action="store_true",
        help="Find class-wise thresholds on validation set and apply them to test set.",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(args.seed)

    save_name = args.experiment

    if args.experiment in [
        "raw1_meta",
        "recon12_meta",
        "raw1_recon12",
        "raw1_recon12_meta",
        "real12_meta",
        "recon12_meta_logit",
        "real12_meta_logit",
        "recon12_meta_feature_gate",
        "real12_meta_feature_gate",
    ]:
        save_name = "{}_{}".format(args.experiment, args.fusion)

    save_dir = os.path.join(args.checkpoint, save_name)
    os.makedirs(save_dir, exist_ok=True)

    print("[INFO] Experiment:", args.experiment)
    print("[INFO] Device:", device)
    print("[INFO] Save dir:", save_dir)

    # In eval-only mode without threshold optimization, only the test set is needed.
    # This avoids unnecessarily loading train/val NPZ files and rebuilding WFDB signals.
    if args.epochs == 0 and not args.optimize_threshold:
        print("[INFO] Eval-only without threshold optimization: loading test set only.")

        testloader, test_ds = build_loader(args, "test")

        trainloader = None
        valloader = None
        train_ds = test_ds
        val_ds = None

        meta_in_dim = test_ds.metadata.shape[1]

    else:
        trainloader, train_ds = build_loader(args, "train")
        valloader, val_ds = build_loader(args, "val")
        testloader, test_ds = build_loader(args, "test")

        meta_in_dim = train_ds.metadata.shape[1]

    model = build_model(args, meta_in_dim=meta_in_dim)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if args.pretrained_wang is not None:
        load_pretrained_wang(model, args.pretrained_wang, device)

        if args.freeze_wang:
            target_model = model.module if isinstance(model, torch.nn.DataParallel) else model

            if hasattr(target_model, "head_feat") and hasattr(target_model, "head_cls"):
                for name, p in target_model.named_parameters():
                    if name.startswith(("stem", "layer1", "layer2", "layer3", "head_feat")):
                        p.requires_grad = False
                print("[INFO] Frozen Wang feature backbone/head parameters.")

            if hasattr(target_model, "ecg_encoder") and hasattr(target_model.ecg_encoder, "backbone"):
                for p in target_model.ecg_encoder.backbone.parameters():
                    p.requires_grad = False
                print("[INFO] Frozen Wang backbone parameters.")

            elif hasattr(target_model, "wang"):
                for p in target_model.wang.parameters():
                    p.requires_grad = False
                print("[INFO] Frozen full Wang classifier parameters.")

    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print("[INFO] Trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("  ", name, tuple(p.shape))

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[INFO] Trainable params: {} / {}".format(n_train, n_total))

    optimizer = None
    if len(trainable_params) > 0:
        optimizer = optim.Adam(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        print("[INFO] No trainable parameters. Running eval-only.")

    best_auc = -1.0
    best_epoch = -1
    history = []

    if optimizer is None or args.epochs == 0:
        print("[INFO] Eval-only mode.")

        threshold = 0.5
        threshold_detail = None

        if args.optimize_threshold:
            val_loss_for_th, y_val_true, y_val_prob = collect_probs(
                valloader, model, criterion, device
            )
            threshold, threshold_detail = find_optimal_thresholds(
                y_val_true,
                y_val_prob,
                metric=args.threshold_metric,
            )

            save_json(
                os.path.join(save_dir, "thresholds.json"),
                {
                    "threshold_metric": args.threshold_metric,
                    "label_names": LABEL_NAMES,
                    "thresholds": {
                        name: float(threshold[i])
                        for i, name in enumerate(LABEL_NAMES)
                    },
                    "detail": threshold_detail,
                    "val_loss_for_threshold_search": float(val_loss_for_th),
                },
            )

            print("[THRESHOLD] metric={}".format(args.threshold_metric))
            print(
                "  " + " | ".join(
                    [
                        "{}={:.2f}".format(name, threshold[i])
                        for i, name in enumerate(LABEL_NAMES)
                    ]
                )
            )

        (
            test_loss,
            test_f1,
            test_auc,
            test_acc,
            test_sens,
            test_spec,
            test_per_label_auc,
            test_per_label_detail,
        ) = evaluate(testloader, model, criterion, device, threshold=threshold)

        test_metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_f1),
            "test_macro_auc": float(test_auc),
            "test_sensitivity": float(test_sens),
            "test_specificity": float(test_spec),
            "test_per_label_auc": test_per_label_auc,
            "test_per_label_detail": test_per_label_detail,
            "best_epoch": 0,
            "best_val_macro_auc": float("nan"),
            "label_names": LABEL_NAMES,
            "optimize_threshold": bool(args.optimize_threshold),
            "threshold_metric": args.threshold_metric if args.optimize_threshold else None,
            "thresholds": (
                {name: float(threshold[i]) for i, name in enumerate(LABEL_NAMES)}
                if args.optimize_threshold else 0.5
            ),
            "threshold_detail": threshold_detail,
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

        return

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

    threshold = 0.5
    threshold_detail = None

    if args.optimize_threshold:
        val_loss_for_th, y_val_true, y_val_prob = collect_probs(
            valloader, model, criterion, device
        )
        threshold, threshold_detail = find_optimal_thresholds(
            y_val_true,
            y_val_prob,
            metric=args.threshold_metric,
        )

        save_json(
            os.path.join(save_dir, "thresholds.json"),
            {
                "threshold_metric": args.threshold_metric,
                "label_names": LABEL_NAMES,
                "thresholds": {
                    name: float(threshold[i])
                    for i, name in enumerate(LABEL_NAMES)
                },
                "detail": threshold_detail,
                "val_loss_for_threshold_search": float(val_loss_for_th),
            },
        )

        print("[THRESHOLD] metric={}".format(args.threshold_metric))
        print(
            "  " + " | ".join(
                [
                    "{}={:.2f}".format(name, threshold[i])
                    for i, name in enumerate(LABEL_NAMES)
                ]
            )
        )

    (
        test_loss,
        test_f1,
        test_auc,
        test_acc,
        test_sens,
        test_spec,
        test_per_label_auc,
        test_per_label_detail,
    ) = evaluate(testloader, model, criterion, device, threshold=threshold)

    test_metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_macro_auc": float(test_auc),
        "test_sensitivity": float(test_sens),
        "test_specificity": float(test_spec),
        "test_per_label_auc": test_per_label_auc,
        "test_per_label_detail": test_per_label_detail,
        "best_epoch": int(best_epoch),
        "best_val_macro_auc": float(best_auc),
        "label_names": LABEL_NAMES,
        "optimize_threshold": bool(args.optimize_threshold),
        "threshold_metric": args.threshold_metric if args.optimize_threshold else None,
        "thresholds": (
            {name: float(threshold[i]) for i, name in enumerate(LABEL_NAMES)}
            if args.optimize_threshold else 0.5
        ),
        "threshold_detail": threshold_detail,
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