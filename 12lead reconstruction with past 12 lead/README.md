# 12-lead ECG Reconstruction with Past 12-lead

현재 **Lead I** + 같은 환자의 **과거 12-lead ECG** → 현재 **12-lead ECG** 복원
(Prior-Conditioned Cascade Model on PTB-XL)

## 📁 Repo 구조

```
.
├── cascade_recon.py            # 모델 정의 (UNet1D, CascadeECGRecon, FiLM)
├── train_cascade_prior.py      # 학습 스크립트 (PriorEncoder + Cascade)
├── infer_cascade_prior.py      # 추론 스크립트
├── requirements.txt
└── outputs/cascade_prior/      # 학습 산출물 (직접 추가)
    ├── best.pt                 # ✅ 학습된 모델 가중치
    ├── config.json             # ✅ 모델 설정 (자동 생성)
    ├── meta_stats.npz          # ✅ 메타 정규화 통계 (use_meta=True일 때)
    ├── test_pairs.csv          # ✅ test split (자동 생성)
    ├── val_pairs.csv
    └── train_pairs.csv
```

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

PTB-XL 데이터셋 (1.0.3) 필요:
<https://physionet.org/content/ptb-xl/1.0.3/>

## 🚂 학습

```bash
python train_cascade_prior.py \
    --data_dir /path/to/ptb-xl/1.0.3/ \
    --output_dir ./outputs/cascade_prior \
    --use_meta \
    --teacher_forcing \
    --epochs 30 \
    --batch_size 32
```

저장되는 파일들:
- `best.pt` / `last.pt` — 모델 가중치
- `config.json` — 모델 hyperparameter (인퍼런스 시 자동 로드)
- `meta_stats.npz` — 키/체중 정규화 통계 (`--use_meta`일 때)
- `train_pairs.csv` / `val_pairs.csv` / `test_pairs.csv` — split 페어

## 🔬 추론

학습이 끝났거나 미리 받은 `best.pt`가 있으면:

```bash
python infer_cascade_prior.py \
    --data_dir /path/to/ptb-xl/1.0.3/ \
    --ckpt_dir ./outputs/cascade_prior \
    --ckpt_name best.pt \
    --save_plots 10
```

결과:
```
outputs/cascade_prior/inference/
├── reconstructions.npz     # inputs, targets, preds, past, time_delta, pairs
├── metrics_per_sample.csv  # 샘플별 MAE/Pearson (lead별)
├── metrics_summary.csv     # 전체 평균/표준편차
└── sample_plots/*.png      # 시각화
```

## 🧪 모델 구조 요약

1. **PriorEncoder**: 과거 12-lead (12,T) → 압축 벡터 (prior_dim-1) + time_delta(1)
2. **(옵션) 환자 메타**: 나이/성별/키/체중 → 6차원
3. **conditioning** = prior_vec ⊕ meta_vec → base 모델의 FiLM에 주입
4. **Cascade**:
   - `gen_II`: Lead I → Lead II
   - 사지 유도(III, aVR, aVL, aVF): 산술 변환 + residual UNet
   - `gen_V`: 6-lead context → V1~V6

## 📦 학습된 모델만 공유할 때 같이 올려야 하는 파일

`best.pt` 만으로는 작동 안 됩니다. 다음 파일들이 **같은 `ckpt_dir`에 있어야** 추론 가능:

| 파일                | 필수? | 역할                                  |
|---------------------|-------|---------------------------------------|
| `best.pt`           | ✅    | 모델 가중치                           |
| `config.json`       | ✅    | base_ch / prior_dim / use_meta 등     |
| `meta_stats.npz`    | △     | `use_meta=True`로 학습했을 때만 필요  |
| `test_pairs.csv`    | △     | 없으면 fold=10에서 자동 재구성        |
