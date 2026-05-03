# ==========================================
# Dockerfile: ECG 1-Lead -> 12-Lead Reconstruction
# ==========================================
# 서버 환경:
#   - GPU: NVIDIA TITAN X (Pascal) x8 (12GB each)
#   - Driver: 510.60.02
#   - CUDA 지원: 11.6 (드라이버 기준 최대)
#   - OS: Ubuntu 16.04
#
# PyTorch 1.13.1 + CUDA 11.6 사용 (드라이버 호환)
# ==========================================

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /workspace

# 의존성 먼저 복사 (Docker 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 코드 복사
COPY ecg_models/models/ ./ecg_models/models/
COPY ecg_1to12/ ./ecg_1to12/
COPY PAPERS.md ./PAPERS.md
COPY run_all.sh ./run_all.sh
COPY inference.py ./inference.py
RUN chmod +x run_all.sh

# 데이터/출력/추론결과 디렉토리 (마운트 포인트)
RUN mkdir -p /workspace/data/ptb-xl
RUN mkdir -p /workspace/outputs
RUN mkdir -p /workspace/results

# 환경 변수
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# 기본: 도움말 표시
CMD ["python", "ecg_1to12/train_all_models.py", "--help"]
