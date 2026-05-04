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

from sklearn.metrics import roc_auc_score

from datasets.ptbxl_dataset import PTBXLMultimodalDataset
from models.resnet_lstm_ptbxl_raw import RawECGEncoder


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]


class MultiLeadRawClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=1,
        feature_dim=128,
        lstm_hidden=128,
        lstm_layers=1,
        layers=(2, 2, 2),
        base_channels=32,
    ):
        super().__init__()

        self.encoder = RawECGEncoder(
            in_channels=in_channels,
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


def seed_everything(seed):
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

    return data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.workers,
        pin_memory=True,
    )


def get_pos_weight(train_dataset, label_idx, device):
    labels = np.stack(train_dataset.df["label_vec"].values)
    y = labels[:, label_idx]

    pos = y.sum()
    neg = len(y) - pos

    if pos == 0:
        return None

    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)


def select_leads_label(batch, selected_leads, label_idx, device):
    x = batch["ecg_raw"][:, selected_leads, :].float().to(device)
    y = batch["label"][:, label_idx:label_idx + 1].float().to(device)
    return x, y


def train_one_epoch(loader, model, criterion, optimizer, device, selected_leads, label_idx):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, leave=False):
        x, y = select_leads_label(batch, selected_leads, label_idx, device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_auc(loader, model, criterion, device, selected_leads, label_idx):
    model.eval()
    total_loss = 0.0

    all_targets = []
    all_probs = []

    for batch in tqdm(loader, leave=False):
        x, y = select_leads_label(batch, selected_leads, label_idx, device)

        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * y.size(0)

        all_targets.append(y.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0).reshape(-1)
    y_prob = np.concatenate(all_probs, axis=0).reshape(-1)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / len(loader.dataset),
        "auroc": auc,
    }


def append_csv(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "label_idx",
        "label_name",
        "lead_setting",
        "selected_lead_indices",
        "selected_lead_names",
        "epochs",
        "best_epoch",
        "best_val_auroc",
        "best_val_loss",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def run_experiment(args, label_idx, device):
    print("=" * 100)
    print(f"12-lead training | Label: {label_idx} ({LABELS[label_idx]})")
    print("=" * 100)

    seed_everything(args.seed)

    selected_leads = list(range(12))

    trainloader = build_dataloader(args, "train")
    valloader = build_dataloader(args, "val")

    model = MultiLeadRawClassifier(
        in_channels=len(selected_leads),
        num_classes=1,
    ).to(device)

    pos_weight = get_pos_weight(trainloader.dataset, label_idx, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_auc = -1.0
    best_epoch = -1
    best_loss = None

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            trainloader,
            model,
            criterion,
            optimizer,
            device,
            selected_leads,
            label_idx,
        )

        val_metrics = evaluate_auc(
            valloader,
            model,
            criterion,
            device,
            selected_leads,
            label_idx,
        )

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['auroc']:.4f}"
        )

        if val_metrics["auroc"] > best_auc:
            best_auc = val_metrics["auroc"]
            best_epoch = epoch + 1
            best_loss = val_metrics["loss"]

    row = {
        "label_idx": label_idx,
        "label_name": LABELS[label_idx],
        "lead_setting": "12-lead",
        "selected_lead_indices": str(selected_leads),
        "selected_lead_names": ",".join([LEADS[i] for i in selected_leads]),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_auroc": best_auc,
        "best_val_loss": best_loss,
    }

    append_csv(args.save_csv, row)

    print("Saved:", args.save_csv)
    print(row)


def main():
    parser = argparse.ArgumentParser(description="PTB-XL 12-lead AUC training")

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])

    parser.add_argument("--label_idx", type=int, default=None)
    parser.add_argument("--run_all_labels", action="store_true")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--save_csv", type=str, default="./results/auc_12lead_results.csv")
    parser.add_argument("--available_ecg_ids", type=int, nargs="*", default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.run_all_labels:
        for label_idx in range(len(LABELS)):
            run_experiment(args, label_idx, device)
    else:
        if args.label_idx is None:
            raise ValueError("Use --label_idx or --run_all_labels")

        run_experiment(args, args.label_idx, device)


if __name__ == "__main__":
    main()