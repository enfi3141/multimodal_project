# ECG 1-Lead → 12-Lead Reconstruction: 논문 및 참조 목록

## 공식 GitHub 코드 기반 모델 (직접 추출)

### 1. UNet (hawkiyc)
- **논문 참고**: "Learning to See in the Dark" (Chen et al., 2018) 에서 영감
- **프로젝트**: 12 Leads ECG Signal Reconstruction from Single Lead
- **GitHub**: https://github.com/hawkiyc/12_Leads_ECG_Reconstruction_wtih_U_Net
- **모델 파일**: `models/unet_hawkiyc.py`
- **원본 위치**: `train.py` (lines 220-296)
- **구조**: 1D U-Net (DoubleConv → Down → Up → OutConv)
- **특징**: kernel_size=19, LeakyReLU, Dropout, Lead II → 11 leads

### 2. GAN (Zehui Zhan)
- **논문**: "Conditional generative adversarial network driven variable-duration single-lead to 12-lead electrocardiogram reconstruction"
- **저자**: Zehui Zhan, Jiarong Chen, Kangming Li, Wanqing Wu
- **GitHub**: https://github.com/Zehui-Zhan/12-lead-reconstruction
- **모델 파일**: `models/gan_zehui.py`
- **원본 위치**: `GAN.py` (Generator_gan: lines 37-85, Discriminator_gan: lines 86-103)
- **구조**: U-Net 스타일 Generator + 6단계 Discriminator
- **특징**: 7단계 encoder + 7단계 decoder with skip connections, WGAN loss

### 3. VAE (Zehui Zhan)
- **논문**: 위 GAN 논문과 동일 (비교 모델로 사용)
- **GitHub**: https://github.com/Zehui-Zhan/12-lead-reconstruction
- **모델 파일**: `models/vae_zehui.py`
- **원본 위치**: `VAE_CNN.py` (VAE class: lines 16-94)
- **구조**: Conv1D Encoder → μ/σ → Reparameterization → Conv1D Decoder
- **특징**: latent_dim=128, [16,32,64,128,256] hidden channels

### 4. CNN-LSTM (Zehui Zhan)
- **논문**: 위 GAN 논문과 동일 (비교 모델로 사용)
- **GitHub**: https://github.com/Zehui-Zhan/12-lead-reconstruction
- **모델 파일**: `models/lstm_zehui.py`
- **원본 위치**: `LSTM.py` (Generator_lstm class: lines 14-40)
- **구조**: Conv1D → MaxPool → Conv1D → MaxPool → Conv1D → MaxPool → LSTM → Dense → Conv1D
- **특징**: LSTM hidden_size=64, Conv kernel_size=5

### 5. ECGrecover (UMMISCO)
- **논문**: "ECGrecover: A Deep Learning Approach for Electrocardiogram Signal Completion"
- **GitHub**: https://github.com/UMMISCO/ecgrecover
- **모델 파일**: `models/ecgrecover_ummisco.py`
- **원본 위치**: `tools/LoadModel.py` (Autoencoder_net: lines 80-179)
- **구조**: 1D + 2D 병렬 Conv를 사용하는 하이브리드 U-Net Autoencoder
- **특징**: 리드 간 관계(2D) + 각 리드 독립 처리(1D) 병렬 결합
- **pretrained**: `model/Model.pth` (24.6MB)

### 6. MCMA UNet3+ (CHENJIAR3)
- **논문**: "Multi-channel masked autoencoder for 12-lead ECG reconstruction from arbitrary single-lead"
- **저널**: npj Cardiovascular Health, 2024
- **GitHub**: https://github.com/CHENJIAR3/MCMA
- **모델 파일**: `models/mcma_unet3plus.py`
- **원본 위치**: `model.py` (downblock, upblock, unet3plus_block, modelx: lines 4-94)
- **구조**: UNet3+ 백본 (6단계 인코더, 5단계 디코더 with dense skip connections)
- **특징**: Dual-path (GELU + LayerNorm), InstanceNorm, kernel_size=13
- **⚠️ 변환**: TensorFlow → PyTorch (구조/파라미터 동일 유지)

---

## 공식 GitHub 미공개 / Clone만 된 참고 저장소

### 7. EKGAN (MICCAI 2023)
- **논문**: "Twelve-lead ECG Reconstruction from Single-lead via Knowledge Distillation" (MICCAI 2023)
- **GitHub**: https://github.com/knu-plml/ecg-recon
- **클론 폴더**: `ekgan_official/`
- **⚠️ TensorFlow/Keras 기반** — PyTorch 래핑 미수행 (2D Conv 기반, 별도 프레임워크 필요)

### 8. ECG-FM (Foundation Model)
- **논문**: "ECG-FM: An Open Electrocardiogram Foundation Model"
- **GitHub**: https://github.com/bowang-lab/ecg-fm
- **클론 폴더**: `ecg_fm_official/`
- **⚠️ fairseq_signals 프레임워크 의존** — 독립 모델 파일 추출 불가

### 9. ODE-ECG (Golany)
- **논문**: "12-Lead ECG Reconstruction via Koopman Operators" (ICML 2021)
- **저자**: Tomer Golany, Kira Radinsky, Daniel Freedman, Saar Minha
- **GitHub**: https://github.com/tomerGolany/ode_ecg
- **클론 폴더**: `koopman_ode_official/`
- **⚠️ ECG 생성(synthesis)용** — 1→12 리드 직접 복원과는 목적이 다름

### 10. Scripps 3→12 Lead (ResCNN)
- **논문**: "AI-Enhanced Reconstruction of the 12-Lead ECG via 3-Leads with Accurate Clinical Assessment"
- **GitHub**: https://github.com/scripps-research/ecg_reconstruction
- **클론 폴더**: `scripps_ecg_recon/`
- **⚠️ 3리드(I, II, V3) → 12리드** — 1리드 입력이 아님

### 11. Kubota CNN-BiLSTM + Metadata
- **논문**: Multi-Lead ECG Reconstruction with Clinical Metadata
- **GitHub**: https://github.com/kubota0728/multi-lead-ecg-reconstruction
- **클론 폴더**: `kubota_ecg_recon/`
- **⚠️ MATLAB 기반** — Python/PyTorch 환경과 직접 호환 불가

---

## 코드 미공개 논문 (Mockup 상태 유지)

| 논문 | 상태 | arXiv / 참조 |
|------|------|-------------|
| mEcgNet (MICCAI 2025) | 코드 미공개 | MICCAI proceedings |
| SelfMIS | 공개 예정 | arXiv:2509.19397 |
| TransECG | 공개 예정 | arXiv:2503.13495 |
| P2Es (MobiCom 2025) | 코드 미공개 | arXiv:2509.25480 |
| Shifted Diffusion ECG | 코드 미공개 | ECML-PKDD 2024 |


코드 내부의 자동 분류:
제가 작성한 코드가 이 엑셀 파일을 딱 읽고, 자동으로 1번~9번 폴더의 환자들(전체의 90%)은 학습(Train) 에 밀어 넣습니다.
그리고 10번 폴더의 환자들(나머지 10%)은 오직 시험용(Valid) 으로만 뺴둡니다. (절대 학습할 때 안 보여줍니다.)