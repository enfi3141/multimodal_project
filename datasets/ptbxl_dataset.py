import ast
import os
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb


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


class PTBXLMultimodalDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        sampling_rate: int = 100,
        split: str = "train",
        folds_train: Optional[List[int]] = None,
        folds_val: Optional[List[int]] = None,
        folds_test: Optional[List[int]] = None,
        image_transform: Optional[Callable] = None,
        use_raw: bool = True,
        use_metadata: bool = True,
        use_image: bool = False,
        precomputed_img_root: Optional[str] = None,
        image_subdir: str = "stft_npy",
        available_ecg_ids: Optional[List[int]] = None,
    ):
        super().__init__()

        self.root_dir = os.path.abspath(root_dir)
        self.sampling_rate = sampling_rate
        self.split = split
        self.image_transform = image_transform
        self.use_raw = use_raw
        self.use_metadata = use_metadata
        self.use_image = use_image
        self.precomputed_img_root = (
            os.path.abspath(precomputed_img_root)
            if precomputed_img_root is not None
            else None
        )
        self.image_subdir = image_subdir

        if folds_train is None:
            folds_train = [1, 2, 3, 4, 5, 6, 7, 8]
        if folds_val is None:
            folds_val = [9]
        if folds_test is None:
            folds_test = [10]

        # 디버그용으로만 사용
        if available_ecg_ids is not None and len(available_ecg_ids) == 0:
            available_ecg_ids = None
        self.available_ecg_ids = available_ecg_ids

        # PTB-XL 루트 확인
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"PTB-XL root directory not found: {self.root_dir}")

        self.db_path = os.path.join(self.root_dir, "ptbxl_database.csv")
        self.scp_path = os.path.join(self.root_dir, "scp_statements.csv")
        self.records100_dir = os.path.join(self.root_dir, "records100")
        self.records500_dir = os.path.join(self.root_dir, "records500")

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"ptbxl_database.csv not found: {self.db_path}\n"
                f"Expected PTB-XL root like: /workspace/data/ptbxl"
            )

        if not os.path.exists(self.scp_path):
            raise FileNotFoundError(
                f"scp_statements.csv not found: {self.scp_path}\n"
                f"Expected PTB-XL root like: /workspace/data/ptbxl"
            )

        if self.sampling_rate == 100 and not os.path.isdir(self.records100_dir):
            raise FileNotFoundError(
                f"records100 directory not found: {self.records100_dir}"
            )

        if self.sampling_rate == 500 and not os.path.isdir(self.records500_dir):
            raise FileNotFoundError(
                f"records500 directory not found: {self.records500_dir}"
            )

        self.df = pd.read_csv(self.db_path)
        self.df["scp_codes"] = self.df["scp_codes"].apply(ast.literal_eval)

        scp_df = pd.read_csv(self.scp_path, index_col=0)
        scp_df = scp_df[scp_df["diagnostic"] == 1]
        self.scp_df = scp_df

        self.df["label_vec"] = self.df["scp_codes"].apply(
            lambda x: aggregate_superclasses(x, self.scp_df)
        )

        if split == "train":
            self.df = self.df[self.df["strat_fold"].isin(folds_train)].reset_index(drop=True)
        elif split == "val":
            self.df = self.df[self.df["strat_fold"].isin(folds_val)].reset_index(drop=True)
        elif split == "test":
            self.df = self.df[self.df["strat_fold"].isin(folds_test)].reset_index(drop=True)
        else:
            raise ValueError("split must be one of ['train', 'val', 'test']")

        if self.available_ecg_ids is not None:
            self.df = self.df[self.df["ecg_id"].isin(self.available_ecg_ids)].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found for split='{split}'. "
                f"Check root_dir={self.root_dir}, strat_fold, "
                f"or available_ecg_ids={self.available_ecg_ids}"
            )

    def __len__(self):
        return len(self.df)

    def _load_signal(self, row) -> np.ndarray:
        if self.sampling_rate == 100:
            rel_path = row["filename_lr"]
        elif self.sampling_rate == 500:
            rel_path = row["filename_hr"]
        else:
            raise ValueError(f"sampling_rate must be 100 or 500, got {self.sampling_rate}")

        record_path = os.path.join(self.root_dir, rel_path)

        # WFDB는 .hea / .dat 쌍을 사용
        hea_path = record_path + ".hea"
        dat_path = record_path + ".dat"

        if not os.path.exists(hea_path):
            raise FileNotFoundError(f"WFDB header file not found: {hea_path}")
        if not os.path.exists(dat_path):
            raise FileNotFoundError(f"WFDB signal file not found: {dat_path}")

        record = wfdb.rdsamp(record_path)[0]      # (time, 12)
        signal = record.T.astype(np.float32)      # (12, time)
        return signal

    def _make_metadata(self, row) -> np.ndarray:
        age = row["age"] if not np.isnan(row["age"]) else 0
        sex = row["sex"] if not np.isnan(row["sex"]) else 0

        age_norm = np.float32(age / 100.0)

        sex_onehot = np.zeros(2, dtype=np.float32)
        if int(sex) == 0:
            sex_onehot[0] = 1.0
        else:
            sex_onehot[1] = 1.0

        metadata = np.concatenate([[age_norm], sex_onehot], axis=0).astype(np.float32)
        return metadata

    def _load_precomputed_image(self, ecg_id: str) -> np.ndarray:
        if self.precomputed_img_root is None:
            raise ValueError("precomputed_img_root is None")

        img_path = os.path.join(
            self.precomputed_img_root,
            self.split,
            self.image_subdir,
            f"{ecg_id}.npy",
        )

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Precomputed image not found: {img_path}")

        ecg_img = np.load(img_path).astype(np.float32)
        return ecg_img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        label = row["label_vec"].astype(np.float32)

        sample = {
            "id": str(row["ecg_id"]),
            "label": torch.from_numpy(label).float(),
        }

        ecg = None
        if self.use_raw or (self.use_image and self.precomputed_img_root is None):
            ecg = self._load_signal(row)

        if self.use_raw:
            sample["ecg_raw"] = torch.from_numpy(ecg).float()

        if self.use_metadata:
            metadata = self._make_metadata(row)
            sample["metadata"] = torch.from_numpy(metadata).float()

        if self.use_image:
            if self.precomputed_img_root is not None:
                ecg_img = self._load_precomputed_image(str(row["ecg_id"]))
            else:
                if self.image_transform is None:
                    raise ValueError("use_image=True but image_transform is None")
                ecg_img = self.image_transform(ecg)

            if not isinstance(ecg_img, np.ndarray):
                raise TypeError("ecg_img must be a numpy.ndarray")

            if ecg_img.ndim != 4:
                raise ValueError(
                    f"Invalid ecg_img shape: {ecg_img.shape}, expected (T, 3, H, W)"
                )

            if ecg_img.shape[1] != 3:
                raise ValueError(
                    f"Invalid ecg_img channel shape: {ecg_img.shape}, second dim must be 3"
                )

            sample["ecg_img"] = torch.from_numpy(ecg_img).float()

        return sample