import ast
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


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

    record = wfdb.rdsamp(path)[0]            # (time, 12)
    signal_arr = record.T.astype(np.float32) # (12, time)
    return signal_arr


def resample_1d(x: np.ndarray, out_len: int) -> np.ndarray:
    """
    1D signal을 선형보간으로 out_len 길이로 변환
    """
    if len(x) == out_len:
        return x.astype(np.float32)

    old_idx = np.linspace(0, len(x) - 1, num=len(x), dtype=np.float32)
    new_idx = np.linspace(0, len(x) - 1, num=out_len, dtype=np.float32)
    y = np.interp(new_idx, old_idx, x).astype(np.float32)
    return y


def normalize_to_minus1_1(x: np.ndarray) -> np.ndarray:
    """
    min-max 정규화 후 [-1, 1]로 변환
    """
    x_min = np.min(x)
    x_max = np.max(x)

    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    x = (x - x_min) / (x_max - x_min)   # [0, 1]
    x = 2.0 * x - 1.0                   # [-1, 1]
    x = np.clip(x, -1.0, 1.0)
    return x.astype(np.float32)


def gaf_transform(
    ecg: np.ndarray,
    lead_idx: int = 0,
    out_size: int = 224,
    method: str = "summation",
) -> np.ndarray:
    """
    ecg: (12, time)
    return: (1, 3, H, W)

    - lead 0 하나 사용
    - 1D signal -> 길이 out_size로 resample
    - GAF 생성
    - 3채널 복제
    """
    sig = ecg[lead_idx].astype(np.float32)  # (time,)

    # 길이 맞추기
    sig = resample_1d(sig, out_size)

    # [-1, 1] 정규화
    sig = normalize_to_minus1_1(sig)

    # 각도 변환
    phi = np.arccos(sig)  # (out_size,)

    if method == "summation":
        # GASF
        gaf = np.cos(phi[:, None] + phi[None, :])
    elif method == "difference":
        # GADF
        gaf = np.sin(phi[:, None] - phi[None, :])
    else:
        raise ValueError("method must be one of ['summation', 'difference']")

    # [-1, 1] -> [0, 1]
    gaf = (gaf + 1.0) / 2.0
    gaf = np.clip(gaf, 0.0, 1.0).astype(np.float32)

    # 3채널 복제: (3, H, W)
    gaf = np.stack([gaf, gaf, gaf], axis=0).astype(np.float32)

    # 시간축 T=1 추가: (1, 3, H, W)
    gaf = np.expand_dims(gaf, axis=0).astype(np.float32)

    return gaf


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
    lead_idx: int,
    out_size: int,
    method: str,
):
    split_dir = Path(out_root) / split
    img_dir = split_dir / "gaf_npy"
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Saving {split}"):
        ecg_id = str(row.ecg_id)

        ecg = load_signal(root_dir, row, sampling_rate)
        gaf_img = gaf_transform(
            ecg,
            lead_idx=lead_idx,
            out_size=out_size,
            method=method,
        )

        out_path = img_dir / f"{ecg_id}.npy"
        np.save(out_path, gaf_img)

        label_vec = row.label_vec.astype(np.float32)

        age = row.age if not np.isnan(row.age) else 0
        sex = row.sex if not np.isnan(row.sex) else 0

        manifest_rows.append({
            "ecg_id": ecg_id,
            "split": split,
            "gaf_path": str(out_path),
            "label_0_NORM": float(label_vec[0]),
            "label_1_MI": float(label_vec[1]),
            "label_2_STTC": float(label_vec[2]),
            "label_3_CD": float(label_vec[3]),
            "label_4_HYP": float(label_vec[4]),
            "age": float(age),
            "sex": int(sex),
            "num_frames": int(gaf_img.shape[0]),
            "channels": int(gaf_img.shape[1]),
            "height": int(gaf_img.shape[2]),
            "width": int(gaf_img.shape[3]),
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(split_dir / "manifest.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Precompute GAF images for PTB-XL")
    parser.add_argument("--root_dir", type=str, required=True, help="PTB-XL root directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--lead_idx", type=int, default=0, help="Which ECG lead to use")
    parser.add_argument("--out_size", type=int, default=224)
    parser.add_argument("--method", type=str, default="summation", choices=["summation", "difference"])
    parser.add_argument("--include_test", action="store_true")
    parser.add_argument("--available_ecg_ids", type=int, nargs="*", default=None)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = build_dataframe(args.root_dir)

    # 테스트용: 특정 ecg_id만 사용
    # if args.available_ecg_ids is not None:
    #     df = df[df["ecg_id"].isin(args.available_ecg_ids)].reset_index(drop=True)

    train_df = split_dataframe(df, "train")
    val_df = split_dataframe(df, "val")

    # 테스트용으로 split이 비면 그냥 현재 df를 재사용
    # if len(train_df) == 0:
    #     train_df = df.copy()

    # if len(val_df) == 0:
    #     val_df = df.copy()

    save_split(
        df=train_df,
        root_dir=args.root_dir,
        out_root=args.out_dir,
        split="train",
        sampling_rate=args.sampling_rate,
        lead_idx=args.lead_idx,
        out_size=args.out_size,
        method=args.method,
    )

    save_split(
        df=val_df,
        root_dir=args.root_dir,
        out_root=args.out_dir,
        split="val",
        sampling_rate=args.sampling_rate,
        lead_idx=args.lead_idx,
        out_size=args.out_size,
        method=args.method,
    )

    if args.include_test:
        test_df = split_dataframe(df, "test")
        if len(test_df) == 0:
            test_df = df.copy()

        save_split(
            df=test_df,
            root_dir=args.root_dir,
            out_root=args.out_dir,
            split="test",
            sampling_rate=args.sampling_rate,
            lead_idx=args.lead_idx,
            out_size=args.out_size,
            method=args.method,
        )

    print("Done.")


if __name__ == "__main__":
    main()