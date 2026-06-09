# make_nopair_real12_npz.py

import os
import ast
import argparse
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


LABEL_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


DROP_META_COLS = [
    # label leakage
    "scp_codes",
    "diagnostic_superclass",
    "diagnostic_class",
    "diagnostic_subclass",
    "statement_codes",
    "report",
    "infarction_stadium1",
    "infarction_stadium2",

    # signal path / identifiers
    "filename_lr",
    "filename_hr",
    "ecg_id",
    "patient_id",

    # split leakage
    "strat_fold",
]


def load_ecg_12lead(data_dir, rel_path, crop_len=1000, normalize=False):
    path = os.path.join(data_dir, rel_path)
    signal, _ = wfdb.rdsamp(path)
    signal = signal.astype(np.float32).T  # (12, T)

    if signal.shape[-1] < crop_len:
        raise ValueError(f"Signal too short: {rel_path}, shape={signal.shape}")

    signal = signal[:, :crop_len]

    if normalize:
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (
            signal.std(axis=1, keepdims=True) + 1e-8
        )

    return signal.astype(np.float32)


def load_superclass_map(data_dir):
    scp_path = os.path.join(data_dir, "scp_statements.csv")
    scp_df = pd.read_csv(scp_path, index_col=0)

    code_to_class = {}

    for code, row in scp_df.iterrows():
        if bool(row.get("diagnostic", False)):
            diagnostic_class = row.get("diagnostic_class", None)
            if diagnostic_class in LABEL_NAMES:
                code_to_class[str(code)] = diagnostic_class

    return code_to_class


def make_label(scp_codes_str, code_to_class):
    try:
        scp_codes = ast.literal_eval(scp_codes_str) if isinstance(scp_codes_str, str) else scp_codes_str
    except Exception:
        scp_codes = {}

    y = np.zeros(len(LABEL_NAMES), dtype=np.float32)

    for code, score in scp_codes.items():
        try:
            if float(score) <= 0:
                continue
        except Exception:
            continue

        code = str(code)

        if code in LABEL_NAMES:
            cls = code
        else:
            cls = code_to_class.get(code, None)

        if cls in LABEL_NAMES:
            y[LABEL_NAMES.index(cls)] = 1.0

    return y


def build_all_meta_df(df):
    meta = df.drop(columns=[c for c in DROP_META_COLS if c in df.columns]).copy()

    if "recording_date" in meta.columns:
        dt = pd.to_datetime(meta["recording_date"], errors="coerce")
        meta["recording_year"] = dt.dt.year
        meta["recording_month"] = dt.dt.month
        meta["recording_day"] = dt.dt.day
        meta = meta.drop(columns=["recording_date"])

    for c in meta.columns:
        if meta[c].dtype == bool:
            meta[c] = meta[c].astype(int)

    num_cols = meta.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in meta.columns if c not in num_cols]

    if len(num_cols) > 0:
        medians = meta[num_cols].median()
        meta[num_cols] = meta[num_cols].fillna(medians)

        means = meta[num_cols].mean()
        stds = meta[num_cols].std().replace(0, 1)
        meta[num_cols] = (meta[num_cols] - means) / (stds + 1e-8)

    for c in cat_cols:
        meta[c] = meta[c].fillna("missing").astype(str)

    if len(cat_cols) > 0:
        meta = pd.get_dummies(meta, columns=cat_cols)

    meta = meta.astype(np.float32)

    return meta


def collect_exclude_paths(npz_paths):
    exclude = set()

    for npz_path in npz_paths:
        if npz_path is None or not os.path.exists(npz_path):
            continue

        z = np.load(npz_path, allow_pickle=True)

        if "pairs" not in z.files:
            continue

        pairs = z["pairs"]

        for p in pairs[:, 0]:
            if str(p) != "":
                exclude.add(str(p))

        for p in pairs[:, 1]:
            if str(p) != "":
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

    code_to_class = load_superclass_map(data_dir)

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

    meta_df = build_all_meta_df(df)
    meta_cols = meta_df.columns.tolist()

    print("[INFO] Metadata dim:", len(meta_cols))

    inputs = []
    preds = []
    targets = []
    labels = []
    metas = []
    pairs = []
    ids = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        current_path = str(row[path_col])

        try:
            ecg12 = load_ecg_12lead(
                data_dir=data_dir,
                rel_path=current_path,
                crop_len=1000,
                normalize=False,
            )
        except Exception as e:
            print("[SKIP] load failed:", current_path, e)
            continue

        if ecg12.shape[0] != 12:
            print("[SKIP] invalid lead shape:", current_path, ecg12.shape)
            continue

        y = make_label(row["scp_codes"], code_to_class)

        if y.sum() == 0:
            print("[SKIP] no superclass label:", current_path)
            continue

        meta_vec = meta_df.loc[idx].values.astype(np.float32)

        lead1 = ecg12[0:1, :]

        inputs.append(lead1)
        preds.append(ecg12)
        targets.append(ecg12)
        labels.append(y)
        metas.append(meta_vec)
        pairs.append(["", current_path, 0])
        ids.append(int(row["ecg_id"]))

    inputs = np.stack(inputs).astype(np.float32)
    preds = np.stack(preds).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    labels = np.stack(labels).astype(np.float32)
    metas = np.stack(metas).astype(np.float32)
    pairs = np.asarray(pairs, dtype=object)
    ids = np.asarray(ids, dtype=np.int64)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        inputs=inputs,
        preds=preds,
        targets=targets,
        labels=labels,
        meta=metas,
        meta_cols=np.asarray(meta_cols, dtype=object),
        pairs=pairs,
        ids=ids,
    )

    print("[SAVED]", output_path)
    print("inputs   :", inputs.shape)
    print("preds    :", preds.shape)
    print("targets  :", targets.shape)
    print("labels   :", labels.shape)
    print("meta     :", metas.shape)
    print("meta_cols:", len(meta_cols))
    print("pairs    :", pairs.shape)
    print("ids      :", ids.shape)


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