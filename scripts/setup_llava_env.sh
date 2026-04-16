#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${LLAVA_ENV_NAME:-llava}"
INSTALL_CUDA_COMPILER="${LLAVA_INSTALL_CUDA_COMPILER:-0}"
SKIP_FLASH_ATTN="${LLAVA_SKIP_FLASH_ATTN:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_llava_env.sh [--env-name NAME] [--skip-flash-attn] [--install-cuda-compiler]

Creates a reproducible LLaVA conda environment from the repo root.

Options:
  --env-name NAME              Conda environment name. Default: llava
  --skip-flash-attn            Skip flash-attn installation.
  --install-cuda-compiler      Install conda cuda-compiler if nvcc is missing.
  -h, --help                   Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --skip-flash-attn)
            SKIP_FLASH_ATTN=1
            shift
            ;;
        --install-cuda-compiler)
            INSTALL_CUDA_COMPILER=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found. Install Miniconda/Anaconda first." >&2
    exit 1
fi

cd "$REPO_ROOT"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.10
fi

conda activate "$ENV_NAME"

if [[ "$INSTALL_CUDA_COMPILER" == "1" ]] && ! command -v nvcc >/dev/null 2>&1; then
    conda install -y -c nvidia cuda-compiler
fi

python -m pip install --upgrade pip
python -m pip install --upgrade "setuptools==80.10.2" wheel
python -m pip install -r requirements/llava-team.txt

if [[ "$SKIP_FLASH_ATTN" != "1" ]]; then
    PIP_NO_BUILD_ISOLATION=1 python -m pip install "flash-attn==2.5.8"
fi

python -m pip check

python - <<'PY'
import importlib.metadata as md

packages = [
    "llava",
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "peft",
    "deepspeed",
    "bitsandbytes",
]

optional = ["flash-attn"]

print("Installed package versions:")
for name in packages:
    print(f"  {name}=={md.version(name)}")

for name in optional:
    try:
        print(f"  {name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name}=<not installed>")
PY

cat <<EOF

LLaVA environment is ready.
Suggested cache settings for shared machines:
  export HF_HOME=\${HF_HOME:-/data/\$USER/huggingface}
  export TRANSFORMERS_CACHE=\${TRANSFORMERS_CACHE:-\$HF_HOME/transformers}

Activate with:
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
EOF
