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


def train_and_eval_candidate(args, trainloader, valloader, selected_leads, label_idx, device):
    seed_everything(args.seed)

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

    best_metrics = None
    best_epoch = -1
    best_auc = -1.0

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

        auc = val_metrics["auroc"]

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"leads={[LEADS[i] for i in selected_leads]} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auc={auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            best_metrics = val_metrics

    return best_epoch, best_metrics


def append_csv(csv_path, row, fieldnames):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


CANDIDATE_FIELDNAMES = [
    "label_idx",
    "label_name",
    "step",
    "added_lead_idx",
    "added_lead_name",
    "selected_lead_indices",
    "selected_lead_names",
    "best_epoch",
    "val_auroc",
    "val_loss",
]

SELECTED_FIELDNAMES = [
    "label_idx",
    "label_name",
    "step",
    "selected_lead_indices",
    "selected_lead_names",
    "added_lead_idx",
    "added_lead_name",
    "best_val_auroc",
    "val_loss",
    "improvement",
]


def run_greedy_for_label(args, label_idx, device):
    print("=" * 100)
    print(f"Greedy Search for Label: {label_idx} ({LABELS[label_idx]})")
    print("=" * 100)

    trainloader = build_dataloader(args, "train")
    valloader = build_dataloader(args, "val")

    selected_leads = []
    remaining_leads = list(range(12))

    best_global_auc = -1.0
    best_global_metrics = None

    for step in range(1, args.max_leads + 1):
        print("\n" + "-" * 100)
        print(f"Step {step} | Current selected leads: {[LEADS[i] for i in selected_leads]}")
        print("-" * 100)

        candidate_results = []

        for lead_idx in remaining_leads:
            candidate_leads = selected_leads + [lead_idx]

            print("\nCandidate:", [LEADS[i] for i in candidate_leads])

            best_epoch, metrics = train_and_eval_candidate(
                args,
                trainloader,
                valloader,
                candidate_leads,
                label_idx,
                device,
            )

            auc = metrics["auroc"]

            candidate_results.append({
                "lead_idx": lead_idx,
                "lead_name": LEADS[lead_idx],
                "candidate_leads": candidate_leads,
                "candidate_lead_names": [LEADS[i] for i in candidate_leads],
                "best_epoch": best_epoch,
                "metrics": metrics,
                "auc": auc,
            })

            candidate_row = {
                "label_idx": label_idx,
                "label_name": LABELS[label_idx],
                "step": step,
                "added_lead_idx": lead_idx,
                "added_lead_name": LEADS[lead_idx],
                "selected_lead_indices": str(candidate_leads),
                "selected_lead_names": ",".join([LEADS[i] for i in candidate_leads]),
                "best_epoch": best_epoch,
                "val_auroc": metrics["auroc"],
                "val_loss": metrics["loss"],
            }

            append_csv(args.candidate_csv, candidate_row, CANDIDATE_FIELDNAMES)

        candidate_results = sorted(
            candidate_results,
            key=lambda x: x["auc"],
            reverse=True,
        )

        best_candidate = candidate_results[0]
        new_auc = best_candidate["auc"]
        improvement = new_auc - best_global_auc

        print("\nBest candidate at this step")
        print("Add lead:", best_candidate["lead_name"])
        print("Selected leads:", best_candidate["candidate_lead_names"])
        print(f"val_auc: {new_auc:.4f}")
        print(f"Improvement: {improvement:.4f}")

        if step > 1 and improvement < args.min_improvement:
            print("Stop greedy search: AUROC improvement is smaller than min_improvement.")
            break

        selected_leads = best_candidate["candidate_leads"]
        remaining_leads.remove(best_candidate["lead_idx"])

        best_global_auc = new_auc
        best_global_metrics = best_candidate["metrics"]

        final_row = {
            "label_idx": label_idx,
            "label_name": LABELS[label_idx],
            "step": step,
            "selected_lead_indices": str(selected_leads),
            "selected_lead_names": ",".join([LEADS[i] for i in selected_leads]),
            "added_lead_idx": best_candidate["lead_idx"],
            "added_lead_name": best_candidate["lead_name"],
            "best_val_auroc": best_global_auc,
            "val_loss": best_global_metrics["loss"],
            "improvement": improvement,
        }

        append_csv(args.save_csv, final_row, SELECTED_FIELDNAMES)

        if len(remaining_leads) == 0:
            break

    print("\nFinal greedy result")
    print("Label:", LABELS[label_idx])
    print("Selected leads:", [LEADS[i] for i in selected_leads])
    print(f"Best val_auc: {best_global_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="PTB-XL greedy lead combination search using AUROC only")

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])

    parser.add_argument("--label_idx", type=int, default=None)
    parser.add_argument("--run_all_labels", action="store_true")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_leads", type=int, default=12)
    parser.add_argument("--min_improvement", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--save_csv", type=str, default="./results/greedy_selected_leads_auc.csv")
    parser.add_argument("--candidate_csv", type=str, default="./results/greedy_candidate_results_auc.csv")

    parser.add_argument("--available_ecg_ids", type=int, nargs="*", default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.run_all_labels:
        for label_idx in range(len(LABELS)):
            run_greedy_for_label(args, label_idx, device)
    else:
        if args.label_idx is None:
            raise ValueError("Use --label_idx or --run_all_labels")

        if not (0 <= args.label_idx < len(LABELS)):
            raise ValueError("--label_idx must be 0~4")

        run_greedy_for_label(args, args.label_idx, device)


if __name__ == "__main__":
    main()
