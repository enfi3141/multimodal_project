# make_nopair_real12_npz.py

import os
import ast
import argparse
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


def load_ecg_12lead(data_dir, rel_path, normalize=True):
    path = os.path.join(data_dir, rel_path)
    signal, meta = wfdb.rdsamp(path)
    signal = signal.astype(np.float32)  # (T, 12)
    signal = signal.T                   # (12, T)

    if normalize:
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (
            signal.std(axis=1, keepdims=True) + 1e-8
        )

    return signal.astype(np.float32)


def collect_exclude_paths(npz_paths):
    exclude = set()

    for npz_path in npz_paths:
        if npz_path is None or not os.path.exists(npz_path):
            continue

        z = np.load(npz_path, allow_pickle=True)

        if "pairs" not in z.files:
            continue

        pairs = z["pairs"]

        # pairs[:, 0] = past_path, pairs[:, 1] = current_path
        for p in pairs[:, 0]:
            exclude.add(str(p))
        for p in pairs[:, 1]:
            exclude.add(str(p))

    return exclude


def make_npz(
    data_dir,
    output_path,
    folds,
    use_hr=False,
    exclude_npz_paths=None,
):
    db_path = os.path.join(data_dir, "ptbxl_database.csv")
    df = pd.read_csv(db_path)

    path_col = "filename_hr" if use_hr else "filename_lr"

    df = df[df["strat_fold"].isin(folds)].copy()

    exclude_paths = collect_exclude_paths(exclude_npz_paths or [])

    if len(exclude_paths) > 0:
        before = len(df)

        lr_mask = df["filename_lr"].astype(str).isin(exclude_paths)
        hr_mask = df["filename_hr"].astype(str).isin(exclude_paths)

        df = df[~(lr_mask | hr_mask)].copy()

        after = len(df)
        print(f"[INFO] Excluded paired records: {before - after}")

    inputs = []
    preds = []
    targets = []
    pairs = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        current_path = str(row[path_col])

        ecg12 = load_ecg_12lead(data_dir, current_path)  # (12, T)

        if use_hr:
            t = ecg12.shape[-1]
            crop_len = 1000

            if t < crop_len:
                print("[SKIP] too short:", current_path, ecg12.shape)
                continue

            start = (t - crop_len) // 2
            end = start + crop_len
            ecg12 = ecg12[:, start:end]

        if ecg12.shape[0] != 12:
            print("[SKIP] invalid lead shape:", current_path, ecg12.shape)
            continue

        lead1 = ecg12[0:1, :]  # (1, T)

        inputs.append(lead1)
        preds.append(ecg12)
        targets.append(ecg12)

        pairs.append(["", current_path, 0])

    inputs = np.stack(inputs).astype(np.float32)
    preds = np.stack(preds).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    pairs = np.asarray(pairs, dtype=object)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        inputs=inputs,
        preds=preds,
        targets=targets,
        pairs=pairs,
    )

    print("[SAVED]", output_path)
    print("inputs :", inputs.shape)
    print("preds  :", preds.shape)
    print("targets:", targets.shape)
    print("pairs  :", pairs.shape)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--use_hr", action="store_true")
    parser.add_argument("--exclude_npz", type=str, nargs="*", default=[])

    args = parser.parse_args()

    make_npz(
        data_dir=args.data,
        output_path=args.output,
        folds=args.folds,
        use_hr=args.use_hr,
        exclude_npz_paths=args.exclude_npz,
    )


if __name__ == "__main__":
    main()