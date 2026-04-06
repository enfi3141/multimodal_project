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
        folds_train: List[int] = [1, 2, 3, 4, 5, 6, 7, 8],
        folds_val: List[int] = [9],
        folds_test: List[int] = [10],
        image_transform: Optional[Callable] = None,
        use_raw: bool = True,
        use_metadata: bool = True,
        use_image: bool = False,
        precomputed_img_root: Optional[str] = None,
        image_subdir: str = "stft_npy",
        available_ecg_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.sampling_rate = sampling_rate
        self.split = split
        self.image_transform = image_transform
        self.use_raw = use_raw
        self.use_metadata = use_metadata
        self.use_image = use_image
        self.precomputed_img_root = precomputed_img_root
        self.image_subdir = image_subdir
        self.available_ecg_ids = available_ecg_ids

        db_path = os.path.join(root_dir, "ptbxl_database.csv")
        scp_path = os.path.join(root_dir, "scp_statements.csv")

        self.df = pd.read_csv(db_path)
        self.df.scp_codes = self.df.scp_codes.apply(ast.literal_eval)

        scp_df = pd.read_csv(scp_path, index_col=0)
        scp_df = scp_df[scp_df.diagnostic == 1]
        self.scp_df = scp_df

        self.df["label_vec"] = self.df.scp_codes.apply(
            lambda x: aggregate_superclasses(x, self.scp_df)
        )

        if split == "train":
            self.df = self.df[self.df.strat_fold.isin(folds_train)].reset_index(drop=True)
        elif split == "val":
            self.df = self.df[self.df.strat_fold.isin(folds_val)].reset_index(drop=True)
        elif split == "test":
            self.df = self.df[self.df.strat_fold.isin(folds_test)].reset_index(drop=True)
        else:
            raise ValueError("split must be one of ['train', 'val', 'test']")

        # 테스트용: 실제 다운로드한 ecg_id만 남기기
        if self.available_ecg_ids is not None:
            self.df = self.df[self.df["ecg_id"].isin(self.available_ecg_ids)].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found for split='{split}'. "
                f"Check strat_fold or available_ecg_ids={self.available_ecg_ids}"
            )

    def __len__(self):
        return len(self.df)

    def _load_signal(self, row) -> np.ndarray:
        if self.sampling_rate == 100:
            path = os.path.join(self.root_dir, row.filename_lr)
        else:
            path = os.path.join(self.root_dir, row.filename_hr)

        record = wfdb.rdsamp(path)[0]          # (time, 12)
        signal = record.T.astype(np.float32)   # (12, time)
        return signal

    def _make_metadata(self, row) -> np.ndarray:
        age = row.age if not np.isnan(row.age) else 0
        sex = row.sex if not np.isnan(row.sex) else 0

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

        ecg = self._load_signal(row)                 # (12, time)
        label = row.label_vec.astype(np.float32)     # (5,)

        sample = {
            "id": str(row.ecg_id),
            "label": torch.from_numpy(label).float(),
        }

        if self.use_raw:
            sample["ecg_raw"] = torch.from_numpy(ecg).float()   # (12, time)

        if self.use_metadata:
            metadata = self._make_metadata(row)                 # (3,)
            sample["metadata"] = torch.from_numpy(metadata).float()

        if self.use_image:
            if self.precomputed_img_root is not None:
                ecg_img = self._load_precomputed_image(str(row.ecg_id))
            else:
                if self.image_transform is None:
                    raise ValueError("use_image=True 인데 image_transform이 없습니다.")
                ecg_img = self.image_transform(ecg)   # 기대: (T, 3, H, W)

            if not isinstance(ecg_img, np.ndarray):
                raise TypeError("ecg_img는 numpy.ndarray여야 합니다.")

            if ecg_img.ndim != 4:
                raise ValueError(
                    f"ecg_img shape가 잘못됨: {ecg_img.shape}, (T, 3, H, W)여야 함"
                )

            if ecg_img.shape[1] != 3:
                raise ValueError(
                    f"ecg_img 채널 수가 잘못됨: {ecg_img.shape}, 두 번째 축은 3이어야 함"
                )

            sample["ecg_img"] = torch.from_numpy(ecg_img).float()

        return sample