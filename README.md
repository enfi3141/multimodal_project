# Cascade ECG 복원: 1-Lead에서 12-Lead로

단일 lead (Lead I) 입력으로부터 12-lead ECG 신호를 복원하는 딥러닝 프레임워크입니다. Cascade 방식과 선택적 메타데이터 컨디셔닝을 활용합니다.

## 개요

본 프로젝트는 Lead I로부터 12-lead ECG 전체를 계층적으로 복원합니다:

```
Lead I → Lead II → 사지유도 (III, aVR, aVL, aVF) → 흉부유도 (V1-V6)
```

**주요 특징:**
- **Cascade 복원**: Einthoven 법칙 기반 산술 prior + Residual learning
- **Teacher forcing**: Scheduled sampling으로 안정적 학습
- **메타데이터 FiLM 컨디셔닝**: 나이, 성별, 키, 몸무게 (+ 결측 플래그)
- **Vanilla baseline 포함**: 비교 실험용
- **Lead별 가중 손실**: 임상적으로 중요한 lead에 집중
- **8 : 1 : 1 데이터 split** (train : val : test)

## 모델 구조

### Cascade 모델

1. **1단계:** Lead I → Lead II (UNet1D + FiLM 컨디셔닝)
2. **2단계:** Einthoven 관계식을 활용한 사지유도 산술 prior 계산:
   - `III = II - I`
   - `aVR = -(I + II) / 2`
   - `aVL = I - II / 2`
   - `aVF = II - I / 2`
3. **3단계:** 개인차 보정을 위한 learned residual 추가
4. **4단계:** 6개 사지유도 → 6개 흉부유도 (UNet1D + FiLM)

### Cascade를 사용한 이유

- 사지유도는 **수학적 관계**가 있어서 활용 가능
- 흉부유도는 풍부한 컨텍스트 필요 (모든 사지유도 + 메타데이터)
- Teacher forcing으로 **오차 전파 제어**

## 설치

### 필수 패키지

```bash
pip install torch pandas numpy wfdb scikit-learn tqdm matplotlib
```

### 데이터

[PTB-XL 데이터셋](https://physionet.org/content/ptb-xl/1.0.3/)을 PhysioNet에서 다운로드하세요.

```
data/
└── ptb-xl/
    └── 1.0.3/
        ├── ptbxl_database.csv
        └── records100/  (또는 records500/ for HR)
            └── ...
```

## 프로젝트 구조

```
.
├── cascade_recon.py     # 모델 정의 (Cascade & Vanilla)
├── train_cascade.py     # 학습 스크립트 (8:1:1 split)
├── infer_cascade.py     # 추론 스크립트 (test set 복원)
└── README.md
```

## 사용법

### 1단계. 학습

학습 시 데이터를 8:1:1로 split하고, 추론에서 재사용할 정보를 자동 저장합니다:
- `best.pt`, `last.pt` — 모델 체크포인트
- `train_recs.csv`, `val_recs.csv`, `test_recs.csv` — split 정보
- `config.json` — 학습 설정
- `meta_stats.npz` — 메타데이터 정규화 통계 (`--use_meta`일 때)

**기본 학습:**
```bash
python train_cascade.py \
  --data_dir /path/to/ptb-xl/1.0.3/ \
  --output_dir ./outputs/cascade \
  --epochs 30 \
  --batch_size 32
```

**메타데이터**
```bash
python train_cascade.py \
  --data_dir /path/to/ptb-xl/1.0.3/ \
  --use_meta \
  --epochs 30 \
  --batch_size 32
```

**Vanilla Baseline (1-lead → 12-lead 직접):**
```bash
python train_cascade.py \
  --data_dir /path/to/ptb-xl/1.0.3/ \
  --vanilla \
  --epochs 30
```

### 2단계. 추론 (test set)

학습 완료 후, 같은 `output_dir`을 가리켜 추론을 실행합니다:

```bash
python infer_cascade.py \
  --data_dir /path/to/ptb-xl/1.0.3/ \
  --ckpt_dir ./outputs/cascade \
  --ckpt_name best.pt \
  --batch_size 32 \
  --save_plots 10
```

**저장되는 결과:**

```
outputs/cascade/inference/
├── reconstructions.npz       # 원본 + 복원 + 입력 신호
├── metrics_per_sample.csv    # 샘플별 lead별 MAE/Pearson
├── metrics_summary.csv       # 전체 평균/표준편차
└── sample_plots/             # 일부 샘플 시각화 (선택)
    ├── records100_*.png
    └── ...
```

### 저장된 결과 활용 예시

**Python에서 npz 불러오기:**

```python
import numpy as np

data = np.load("outputs/cascade/inference/reconstructions.npz", allow_pickle=True)
inputs  = data["inputs"]   # (N, 1, L)  Lead I만
targets = data["targets"]  # (N, 12, L) 원본 12-lead
preds   = data["preds"]    # (N, 12, L) 복원 12-lead
paths   = data["paths"]    # (N,)       원본 파일 경로
leads   = data["leads"]    # ["I","II","III","aVR","aVL","aVF","V1",..,"V6"]
```

**지표 csv 분석:**

```python
import pandas as pd

# 샘플별 lead별 성능
df = pd.read_csv("outputs/cascade/inference/metrics_per_sample.csv")

# 전체 요약
summary = pd.read_csv("outputs/cascade/inference/metrics_summary.csv", index_col=0)
print(summary)
```

## 학습 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--data_dir` | `/workspace/.../ptb-xl/1.0.3/` | PTB-XL 데이터셋 경로 |
| `--output_dir` | `./outputs/cascade` | 체크포인트/split 저장 경로 |
| `--use_hr` | False | 100Hz 대신 500Hz 신호 사용 |
| `--epochs` | 30 | 학습 epoch 수 |
| `--batch_size` | 32 | 배치 크기 |
| `--lr` | 1e-3 | 학습률 |
| `--vanilla` | False | Vanilla UNet baseline 사용 |
| `--base_ch` | 32 | 메인 UNet의 base channel |
| `--res_base_ch` | 16 | Residual UNet의 base channel |
| `--residual_scale` | 0.3 | Residual 보정 스케일 |
| `--teacher_forcing` | False | Scheduled sampling 활성화 |
| `--tf_decay_ratio` | 0.5 | TF 확률 감소 스케줄 |
| `--use_meta` | False | 메타데이터 사용 |
| `--subsample` | 1.0 | 데이터 일부만 사용 (디버깅용) |
| `--seed` | 42 | 랜덤 시드 |

## 추론 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--data_dir` | `/workspace/.../ptb-xl/1.0.3/` | PTB-XL 데이터셋 경로 |
| `--ckpt_dir` | `./outputs/cascade` | 학습 결과 폴더 |
| `--ckpt_name` | `best.pt` | 로드할 체크포인트 |
| `--out_dir` | `<ckpt_dir>/inference` | 추론 결과 저장 폴더 |
| `--batch_size` | 32 | 배치 크기 |
| `--save_plots` | 10 | 시각화할 샘플 수 (0이면 안 그림) |

## 모델 세부사항

### 손실 함수의 Lead 가중치

임상적으로 중요한 lead에 더 높은 가중치 적용:

| Lead | 가중치 |
|---|---|
| II | 2.0 |
| III, aVF | 3.0 |
| V2, V4, V5 | 2.0 |
| 그 외 | 1.0 |

### 메타데이터 특징 (`--use_meta` 사용 시)

6차원 벡터:
- 나이 (100으로 정규화)
- 성별 (이진값)
- 키 (z-score, **학습셋 통계 기준**)
- 몸무게 (z-score, **학습셋 통계 기준**)
- 키 결측 플래그
- 몸무게 결측 플래그

결측값은 학습셋 평균으로 imputation되며, 결측 플래그를 추가해 모델이 결측 정보를 학습할 수 있도록 했습니다. **Val/Test는 학습셋 통계를 그대로 사용**하여 정보 leak을 방지합니다.

## 데이터 split

- **Train : Val : Test = 8 : 1 : 1**
- 동일 `--seed`로 학습/추론 간 split 일관성 보장
- split 정보(`*_recs.csv`)가 저장되므로 학습 후 추론을 다시 돌려도 같은 test set 사용

## 한계점

- **1-lead 복원은 본질적으로 ill-posed 문제**: Lead I만으로는 흉부유도의 공간 정보를 완전히 복원할 수 없음
- Lead별로 복원 품질 차이가 큼 — 사지유도는 Einthoven 법칙 덕분에 잘 되지만 흉부유도는 어려움
- 성능은 학습 데이터의 다양성에 의존

## 데이터셋 인용

PTB-XL을 사용한 경우 다음을 인용해 주세요:

```bibtex
@article{wagner2020ptbxl,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and others},
  journal={Scientific Data},
  year={2020}
}
```

## 라이센스

[여기에 라이센스 정보 추가]

## 감사의 글

- PhysioNet의 PTB-XL 데이터셋
- PyTorch로 구현