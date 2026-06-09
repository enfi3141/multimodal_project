# add_meta_to_recon_npz.py

import os
import argparse
import numpy as np
import pandas as pd


DROP_META_COLS = [
    # label leakage
    "scp_codes",
    "diagnostic_superclass",
    "diagnostic_class",
    "diagnostic_subclass",
    "statement_codes",

    # signal path / identifiers
    "filename_lr",
    "filename_hr",
    "ecg_id",
    "patient_id",

    # split leakage
    "strat_fold",
]


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

    for c in cat_cols:
        meta[c] = meta[c].fillna("missing").astype(str)

    if len(cat_cols) > 0:
        meta = pd.get_dummies(meta, columns=cat_cols)

    return meta.astype(np.float32)


def add_meta_to_recon_npz(data_dir, input_npz, reference_npz, output_npz):
    z = np.load(input_npz, allow_pickle=True)

    if "pairs" not in z.files:
        raise ValueError("Input NPZ must contain 'pairs'.")

    ref = np.load(reference_npz, allow_pickle=True)

    if "meta_cols" not in ref.files:
        raise ValueError("Reference NPZ must contain 'meta_cols'.")

    meta_cols = ref["meta_cols"].tolist()

    current_paths = z["pairs"][:, 1].astype(str)

    db_path = os.path.join(data_dir, "ptbxl_database.csv")
    df = pd.read_csv(db_path)

    lr_set = set(df["filename_lr"].astype(str))
    hr_set = set(df["filename_hr"].astype(str))

    if current_paths[0] in lr_set:
        df = df.set_index("filename_lr")
        print("[INFO] Matched current paths with filename_lr")
    elif current_paths[0] in hr_set:
        df = df.set_index("filename_hr")
        print("[INFO] Matched current paths with filename_hr")
    else:
        raise ValueError(f"Current path not found in PTB-XL database: {current_paths[0]}")

    missing = [p for p in current_paths if p not in df.index]

    if len(missing) > 0:
        raise ValueError(f"Missing paths in ptbxl_database.csv: {missing[:5]}")

    rows = df.loc[current_paths].copy()

    meta_df = build_all_meta_df(rows)
    meta_df = meta_df.reindex(columns=meta_cols, fill_value=0)

    meta = meta_df.values.astype(np.float32)

    save_dict = {k: z[k] for k in z.files}
    save_dict["meta"] = meta
    save_dict["meta_cols"] = np.asarray(meta_cols, dtype=object)

    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, **save_dict)

    print("[SAVED]", output_npz)
    print("input keys :", z.files)
    print("meta       :", meta.shape)
    print("meta_cols  :", len(meta_cols))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--reference_npz", type=str, required=True)
    parser.add_argument("--output_npz", type=str, required=True)

    args = parser.parse_args()

    add_meta_to_recon_npz(
        data_dir=args.data,
        input_npz=args.input_npz,
        reference_npz=args.reference_npz,
        output_npz=args.output_npz,
    )


if __name__ == "__main__":
    main()