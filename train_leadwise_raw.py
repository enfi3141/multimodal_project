from __future__ import print_function

import os
import csv
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from datasets.ptbxl_dataset import PTBXLMultimodalDataset
from models.resnet_lstm_ptbxl_raw import RawECGEncoder


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]


class LeadWiseRawClassifier(nn.Module):
    def __init__(
        self,
        num_classes=1,
        feature_dim=128,
        lstm_hidden=128,
        lstm_layers=1,
        layers=(2, 2, 2),
        base_channels=32,
    ):
        super().__init__()

        self.encoder = RawECGEncoder(
            in_channels=1,
            feature_dim=feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            layers=layers,
            base_channels=base_channels,
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(args, split):
    dataset = PTBXLMultimodalDataset(
        root_dir=args.data,
        sampling_rate=args.sampling_rate,
        split=split,
        use_raw=True,
        use_metadata=False,
        use_image=False,
        available_ecg_ids=args.available_ecg_ids,
    )

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.workers,
        pin_memory=True,
    )
    return loader


def get_pos_weight(train_dataset, label_idx, device):
    labels = np.stack(train_dataset.df["label_vec"].values)
    y = labels[:, label_idx]

    pos = y.sum()
    neg = len(y) - pos

    if pos == 0:
        return None

    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    return pos_weight


def select_lead_label(batch, lead_idx, label_idx, device):
    x = batch["ecg_raw"][:, lead_idx:lead_idx + 1, :].float().to(device)
    y = batch["label"][:, label_idx:label_idx + 1].float().to(device)
    return x, y


def train_one_epoch(loader, model, criterion, optimizer, device, lead_idx, label_idx):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, leave=False):
        x, y = select_lead_label(batch, lead_idx, label_idx, device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(loader, model, criterion, device, lead_idx, label_idx):
    model.eval()
    total_loss = 0.0

    all_targets = []
    all_probs = []

    for batch in tqdm(loader, leave=False):
        x, y = select_lead_label(batch, lead_idx, label_idx, device)

        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * y.size(0)

        all_targets.append(y.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0).reshape(-1)
    y_prob = np.concatenate(all_probs, axis=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    confidence = np.where(y_pred == 1, y_prob, 1.0 - y_prob)
    correct = (y_pred == y_true)

    mean_conf = float(np.mean(confidence))
    correct_conf = float(np.mean(confidence[correct])) if correct.sum() > 0 else float("nan")
    wrong_conf = float(np.mean(confidence[~correct])) if (~correct).sum() > 0 else float("nan")

    avg_loss = total_loss / len(loader.dataset)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1": f1,
        "auroc": auc,
        "mean_confidence": mean_conf,
        "correct_confidence": correct_conf,
        "wrong_confidence": wrong_conf,
        "positive_rate": float(np.mean(y_true)),
    }


def append_result_csv(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "lead_idx",
        "lead_name",
        "label_idx",
        "label_name",
        "epochs",
        "best_epoch",
        "best_val_f1",
        "best_val_auroc",
        "best_val_accuracy",
        "best_val_loss",
        "mean_confidence",
        "correct_confidence",
        "wrong_confidence",
        "positive_rate",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def run_single_experiment(args, lead_idx, label_idx, device):
    print("=" * 80)
    print(f"Lead: {lead_idx} ({LEADS[lead_idx]}) | Label: {label_idx} ({LABELS[label_idx]})")
    print("=" * 80)

    seed_everything(args.seed)

    trainloader = build_dataloader(args, "train")
    valloader = build_dataloader(args, "val")

    model = LeadWiseRawClassifier(num_classes=1).to(device)

    pos_weight = get_pos_weight(trainloader.dataset, label_idx, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    best_epoch = -1
    best_metrics = None

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            trainloader,
            model,
            criterion,
            optimizer,
            device,
            lead_idx,
            label_idx,
        )

        val_metrics = evaluate(
            valloader,
            model,
            criterion,
            device,
            lead_idx,
            label_idx,
        )

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auroc']:.4f} "
            f"wrong_conf={val_metrics['wrong_confidence']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            best_metrics = val_metrics

    row = {
        "lead_idx": lead_idx,
        "lead_name": LEADS[lead_idx],
        "label_idx": label_idx,
        "label_name": LABELS[label_idx],
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_f1": best_metrics["f1"],
        "best_val_auroc": best_metrics["auroc"],
        "best_val_accuracy": best_metrics["accuracy"],
        "best_val_loss": best_metrics["loss"],
        "mean_confidence": best_metrics["mean_confidence"],
        "correct_confidence": best_metrics["correct_confidence"],
        "wrong_confidence": best_metrics["wrong_confidence"],
        "positive_rate": best_metrics["positive_rate"],
    }

    append_result_csv(args.save_csv, row)

    print(f"Saved result to: {args.save_csv}")
    print(row)


def main():
    parser = argparse.ArgumentParser(description="PTB-XL lead-wise raw binary training")

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])

    parser.add_argument("--lead_idx", type=int, default=None)
    parser.add_argument("--label_idx", type=int, default=None)
    parser.add_argument("--run_all", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--save_csv", type=str, default="./results/leadwise_raw_results.csv")

    parser.add_argument("--available_ecg_ids", type=int, nargs="*", default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.run_all:
        for lead_idx in range(12):
            for label_idx in range(5):
                run_single_experiment(args, lead_idx, label_idx, device)
    else:
        if args.lead_idx is None or args.label_idx is None:
            raise ValueError("Use --lead_idx and --label_idx, or use --run_all")

        if not (0 <= args.lead_idx < 12):
            raise ValueError("--lead_idx must be 0~11")

        if not (0 <= args.label_idx < 5):
            raise ValueError("--label_idx must be 0~4")

        run_single_experiment(args, args.lead_idx, args.label_idx, device)


if __name__ == "__main__":
    main()