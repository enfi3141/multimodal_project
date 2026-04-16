#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${LLAVA_DOCKER_IMAGE:-llava-team:latest}"
CACHE_DIR="${LLAVA_HF_CACHE:-$REPO_ROOT/.hf-cache}"
BUILD_FIRST=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_llava_docker.sh [--build] [-- command...]

Runs an interactive GPU Docker container for this repo.

Options:
  --build      Build the image before running.
  --           Pass the remaining command to docker instead of launching bash.
  -h, --help   Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            BUILD_FIRST=1
            shift
            ;;
        --)
            shift
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

mkdir -p "$CACHE_DIR"

if [[ "$BUILD_FIRST" == "1" ]]; then
    docker build -f "$REPO_ROOT/Dockerfile.team" -t "$IMAGE_NAME" "$REPO_ROOT"
fi

if [[ $# -eq 0 ]]; then
    set -- bash
fi

docker run --rm -it \
    --gpus all \
    --ipc=host \
    -v "$REPO_ROOT:/workspace/LLaVA" \
    -v "$CACHE_DIR:/opt/huggingface" \
    -e HF_HOME=/opt/huggingface \
    -e TRANSFORMERS_CACHE=/opt/huggingface/transformers \
    "$IMAGE_NAME" \
    "$@"
