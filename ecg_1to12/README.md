# PTB-XL 1-Lead -> 12-Lead Reconstruction

This folder contains a minimal PyTorch training pipeline for ECG reconstruction.

- Input: single lead (default: Lead I)
- Output: all 12 leads
- Dataset: PTB-XL (auto-download via `wfdb`)

By default, if PTB-XL `strat_fold` exists, validation uses fold 10.

## 1) Enter your running Docker container

```bash
docker exec -it heart_project bash
```

Check GPU inside container:

```bash
nvidia-smi
```

## 2) Move to this project folder

```bash
cd /path/to/your/workspace/ecg_1to12
```

## 3) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Quick smoke run (small subset)

Use low-rate PTB-XL (100 Hz) for a quick check:

```bash
python train_1to12_ptbxl.py \
  --data_dir ./data/ptb-xl \
  --output_dir ./outputs/smoke \
  --epochs 2 \
  --batch_size 16 \
  --max_records 800 \
  --num_workers 4
```

## 5) Main run (12GB GPU recommended start)

Use high-rate PTB-XL (500 Hz):

```bash
python train_1to12_ptbxl.py \
  --data_dir ./data/ptb-xl \
  --output_dir ./outputs/main_hr \
  --use_hr \
  --val_folds 10 \
  --epochs 20 \
  --batch_size 8 \
  --max_records 10000 \
  --num_workers 4
```

If GPU OOM occurs, reduce `--batch_size` to `4`.

## 6) Outputs

Checkpoints are saved at:

- `outputs/.../best.pt`
- `outputs/.../last.pt`

The training log prints:

- `val_mae_no_input` (lower is better)
- `val_pearson_no_input` (higher is better)

## Notes

- Lead order is fixed as `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.
- The input lead can be changed with `--input_lead`.
- Validation fold can be changed with `--val_folds` (for example `9,10`).
- This is a starter baseline for your project and can be extended with STFT loss and multimodal fusion.
