# Team Setup

This project keeps the official LLaVA installation flow and adds the compatibility pins that were required to make the environment stable in this repository.

## Validated package pins

- `python=3.10`
- `torch==2.1.2`
- `torchvision==0.16.2`
- `transformers==4.37.2`
- `accelerate==0.21.0`
- `deepspeed==0.12.6`
- `bitsandbytes`
- `peft==0.7.1`
- `setuptools==80.10.2`
- `flash-attn==2.5.8`

The last three pins are the important compatibility fixes:

- `setuptools==80.10.2`: newer builds dropped `pkg_resources`, which broke `flash-attn`.
- `peft==0.7.1`: newer `peft` versions were incompatible with the official `accelerate==0.21.0` stack.
- `flash-attn==2.5.8`: newer versions did not build cleanly against the official `torch==2.1.2` environment used here.

## Option 1: Conda

From the repo root:

```bash
bash scripts/setup_llava_env.sh
```

If `nvcc` is missing on the machine:

```bash
bash scripts/setup_llava_env.sh --install-cuda-compiler
```

If you only need inference and want to skip `flash-attn`:

```bash
bash scripts/setup_llava_env.sh --skip-flash-attn
```

Then activate the environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llava
```

## Option 2: Docker

Build the image:

```bash
docker build -f Dockerfile.team -t llava-team:latest .
```

Run it with GPU access:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -v "$PWD:/workspace/LLaVA" \
  -v "$PWD/.hf-cache:/opt/huggingface" \
  -e HF_HOME=/opt/huggingface \
  -e TRANSFORMERS_CACHE=/opt/huggingface/transformers \
  llava-team:latest bash
```

Or use the helper:

```bash
bash scripts/run_llava_docker.sh --build
```

## Option 3: VS Code Dev Container

The existing `.devcontainer` setup now calls the same pinned install script:

```bash
LLAVA_INSTALL_CUDA_COMPILER=1 bash ./scripts/setup_llava_env.sh
```

So dev container users and local conda users land on the same package set.

## Cache location

Model downloads can be large. On shared machines, point caches to a larger disk if possible.

```bash
export HF_HOME=/data/$USER/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

## Quick verification

```bash
conda run -n llava python -m llava.eval.run_llava --help
conda run -n llava python scripts/text_image_chat.py --help
```
