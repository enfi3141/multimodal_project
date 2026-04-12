from __future__ import print_function

import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import f1_score, roc_auc_score

import models
from datasets.ptbxl_dataset import PTBXLMultimodalDataset


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_stft_transform(ecg, fs=100, window_size=200, stride=100, out_size=224):
    import cv2
    from scipy import signal

    ecg_1lead = ecg[0]   # (time,)

    T = (len(ecg_1lead) - window_size) // stride + 1
    images = []

    for i in range(T):
        start = i * stride
        segment = ecg_1lead[start:start + window_size]

        f, t, Zxx = signal.stft(segment, fs=fs, nperseg=64)
        spec = np.abs(Zxx)
        spec = np.log1p(spec)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        spec = cv2.resize(spec, (out_size, out_size))

        spec = np.stack([spec, spec, spec], axis=0)  # (3, H, W)
        images.append(spec)

    images = np.stack(images, axis=0).astype(np.float32)  # (T, 3, H, W)
    return images


def build_dataloader(args, split):
    effective_split = split

    dataset = PTBXLMultimodalDataset(
        root_dir=args.data,
        sampling_rate=args.sampling_rate,
        split=effective_split,   # 여기 수정
        use_raw=args.use_raw,
        use_metadata=args.use_metadata,
        use_image=args.use_image,
        image_transform=(
            None if args.precomputed_img_root is not None
            else (simple_stft_transform if args.use_image else None)
        ),
        precomputed_img_root=args.precomputed_img_root,
        image_subdir=args.image_subdir,
        available_ecg_ids=args.available_ecg_ids,   # 여기 수정
    )

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.workers,
        pin_memory=True,
    )
    return loader


def move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def forward_by_mode(model, batch, mode):
    if mode == "image":
        return model(batch["ecg_img"])
    elif mode == "raw":
        return model(batch["ecg_raw"])
    elif mode == "multimodal":
        return model(batch)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.float32)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        macro_auc = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        macro_auc = float("nan")

    return macro_f1, macro_auc


def train_one_epoch(loader, model, criterion, optimizer, device, mode):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)

        targets = batch["label"].float()
        outputs = forward_by_mode(model, batch, mode)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(loader, model, criterion, device, mode):
    model.eval()
    total_loss = 0.0

    all_targets = []
    all_probs = []

    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)

        targets = batch["label"].float()
        outputs = forward_by_mode(model, batch, mode)

        loss = criterion(outputs, targets)
        probs = torch.sigmoid(outputs)

        total_loss += loss.item() * targets.size(0)

        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    macro_f1, macro_auc = compute_metrics(y_true, y_prob)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, macro_f1, macro_auc


def main():
    parser = argparse.ArgumentParser(description="PTB-XL training")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--checkpoint", type=str, default="checkpoint_ptbxl")
    parser.add_argument("--mode", type=str, default="image", choices=["image", "raw", "multimodal"])
    parser.add_argument("--available_ecg_ids", type=int, nargs="*", default=None)

    parser.add_argument("--use_raw", action="store_true")
    parser.add_argument("--use_metadata", action="store_true")
    parser.add_argument("--use_image", action="store_true")

    parser.add_argument("--precomputed_img_root", type=str, default=None)
    parser.add_argument("--image_subdir", type=str, default="stft_npy")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoint, exist_ok=True)
    seed_everything(args.seed)

    trainloader = build_dataloader(args, split="train")
    valloader = build_dataloader(args, split="val")

    if args.mode == "image":
        model = models.__dict__["resnet_lstm_ptbxl_image"](depth=20, num_classes=5)
    elif args.mode == "raw":
        model = models.__dict__["resnet_lstm_ptbxl_raw"](num_classes=5)
    elif args.mode == "multimodal":
        model = models.__dict__["ptbxl_multimodal_net"](num_classes=5)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(trainloader, model, criterion, optimizer, device, args.mode)
        val_loss, val_f1, val_auc = evaluate(valloader, model, criterion, device, args.mode)

        print(
            f"[Epoch {epoch+1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_macro_f1={val_f1:.4f} "
            f"val_macro_auc={val_auc:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "args": vars(args),
                },
                os.path.join(args.checkpoint, "best_model.pth"),
            )

    print(f"Best macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()