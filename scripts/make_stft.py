import ast
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from scipy import signal


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def aggregate_superclasses(scp_codes: dict, scp_df: pd.DataFrame) -> np.ndarray:
    label_vec = np.zeros(len(SUPERCLASSES), dtype=np.float32)

    for code in scp_codes.keys():
        if code not in scp_df.index:
            continue
        diagnostic_class = scp_df.loc[code, "diagnostic_class"]
        if diagnostic_class in SUPERCLASSES:
            label_vec[SUPERCLASSES.index(diagnostic_class)] = 1.0

    return label_vec


def load_signal(root_dir: str, row: pd.Series, sampling_rate: int) -> np.ndarray:
    if sampling_rate == 100:
        path = os.path.join(root_dir, row.filename_lr)
    else:
        path = os.path.join(root_dir, row.filename_hr)

    record = wfdb.rdsamp(path)[0]          # (time, 12)
    signal_arr = record.T.astype(np.float32)   # (12, time)
    return signal_arr


def simple_stft_transform(
    ecg: np.ndarray,
    fs: int = 100,
    window_size: int = 200,
    stride: int = 100,
    out_size: int = 224,
    lead_idx: int = 0,
) -> np.ndarray:
    """
    ecg: (12, time)
    return: (T, 3, H, W)
    """
    ecg_1lead = ecg[lead_idx]  # (time,)

    if len(ecg_1lead) < window_size:
        pad_width = window_size - len(ecg_1lead)
        ecg_1lead = np.pad(ecg_1lead, (0, pad_width), mode="constant")

    T = (len(ecg_1lead) - window_size) // stride + 1
    images = []

    for i in range(T):
        start = i * stride
        segment = ecg_1lead[start:start + window_size]

        _, _, Zxx = signal.stft(
            segment,
            fs=fs,
            nperseg=64,
            noverlap=32,
            boundary=None
        )

        spec = np.abs(Zxx)
        spec = np.log1p(spec)

        spec_min = spec.min()
        spec_max = spec.max()
        spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)

        spec = cv2.resize(spec, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        spec = np.stack([spec, spec, spec], axis=0)  # (3, H, W)
        images.append(spec)

    images = np.stack(images, axis=0).astype(np.float32)  # (T, 3, H, W)
    return images


def build_dataframe(root_dir: str) -> pd.DataFrame:
    db_path = os.path.join(root_dir, "ptbxl_database.csv")
    scp_path = os.path.join(root_dir, "scp_statements.csv")

    df = pd.read_csv(db_path)
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    scp_df = pd.read_csv(scp_path, index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]

    df["label_vec"] = df.scp_codes.apply(lambda x: aggregate_superclasses(x, scp_df))
    return df


def split_dataframe(
    df: pd.DataFrame,
    split: str,
    folds_train=(1, 2, 3, 4, 5, 6, 7, 8),
    folds_val=(9,),
    folds_test=(10,),
) -> pd.DataFrame:
    if split == "train":
        return df[df.strat_fold.isin(folds_train)].reset_index(drop=True)
    elif split == "val":
        return df[df.strat_fold.isin(folds_val)].reset_index(drop=True)
    elif split == "test":
        return df[df.strat_fold.isin(folds_test)].reset_index(drop=True)
    else:
        raise ValueError("split must be one of ['train', 'val', 'test']")


def save_split(
    df: pd.DataFrame,
    root_dir: str,
    out_root: str,
    split: str,
    sampling_rate: int,
    window_size: int,
    stride: int,
    out_size: int,
    lead_idx: int,
):
    split_dir = Path(out_root) / split
    img_dir = split_dir / "stft_npy"
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Saving {split}"):
        ecg_id = str(row.ecg_id)

        ecg = load_signal(root_dir, row, sampling_rate)
        stft_img = simple_stft_transform(
            ecg,
            fs=sampling_rate,
            window_size=window_size,
            stride=stride,
            out_size=out_size,
            lead_idx=lead_idx,
        )

        out_path = img_dir / f"{ecg_id}.npy"
        np.save(out_path, stft_img)

        label_vec = row.label_vec.astype(np.float32)

        age = row.age if not np.isnan(row.age) else 0
        sex = row.sex if not np.isnan(row.sex) else 0

        manifest_rows.append({
            "ecg_id": ecg_id,
            "split": split,
            "stft_path": str(out_path),
            "label_0_NORM": float(label_vec[0]),
            "label_1_MI": float(label_vec[1]),
            "label_2_STTC": float(label_vec[2]),
            "label_3_CD": float(label_vec[3]),
            "label_4_HYP": float(label_vec[4]),
            "age": float(age),
            "sex": int(sex),
            "num_frames": int(stft_img.shape[0]),
            "channels": int(stft_img.shape[1]),
            "height": int(stft_img.shape[2]),
            "width": int(stft_img.shape[3]),
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(split_dir / "manifest.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Precompute STFT images for PTB-XL")
    parser.add_argument("--root_dir", type=str, required=True, help="PTB-XL root directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--out_size", type=int, default=224)
    parser.add_argument("--lead_idx", type=int, default=0, help="Which ECG lead to use")
    parser.add_argument("--include_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = build_dataframe(args.root_dir)

    train_df = split_dataframe(df, "train")
    val_df = split_dataframe(df, "val")

    save_split(
        df=train_df,
        root_dir=args.root_dir,
        out_root=args.out_dir,
        split="train",
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        stride=args.stride,
        out_size=args.out_size,
        lead_idx=args.lead_idx,
    )

    save_split(
        df=val_df,
        root_dir=args.root_dir,
        out_root=args.out_dir,
        split="val",
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        stride=args.stride,
        out_size=args.out_size,
        lead_idx=args.lead_idx,
    )

    if args.include_test:
        test_df = split_dataframe(df, "test")
        save_split(
            df=test_df,
            root_dir=args.root_dir,
            out_root=args.out_dir,
            split="test",
            sampling_rate=args.sampling_rate,
            window_size=args.window_size,
            stride=args.stride,
            out_size=args.out_size,
            lead_idx=args.lead_idx,
        )

    print("Done.")


if __name__ == "__main__":
    main()