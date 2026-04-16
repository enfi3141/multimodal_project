# README for LLaVA Setup

이 저장소에서 LLaVA 환경을 맞추고, 간단한 이미지 추론을 실행하는 최소 절차만 정리한 문서입니다.

## 1. 환경 세팅

저장소 루트에서 아래 커맨드를 실행합니다.

```bash
bash scripts/setup_llava_env.sh
```

`nvcc`가 없는 머신이면 아래처럼 CUDA compiler 설치까지 같이 진행할 수 있습니다.

```bash
bash scripts/setup_llava_env.sh --install-cuda-compiler
```

추론만 빠르게 확인할 목적이면 `flash-attn` 설치를 건너뛸 수도 있습니다.

```bash
bash scripts/setup_llava_env.sh --skip-flash-attn
```

설치가 끝나면 환경을 활성화합니다.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llava
```

## 2. 추론 실행 예시

예제 이미지를 사용해서 바로 한 번 돌려보려면:

```bash
python scripts/text_image_chat.py \
  --model-path liuhaotian/llava-v1.5-7b \
  --image-file llava/serve/examples/extreme_ironing.jpg \
  --prompt "Describe this image in detail."
```

직접 가진 이미지로 테스트하려면 `--image-file`만 바꿔서 실행하면 됩니다.

```bash
python scripts/text_image_chat.py \
  --model-path liuhaotian/llava-v1.5-7b \
  --image-file /path/to/your_image.jpg \
  --prompt "What do you see in this medical image?"
```

## 3. 확인용 도움말

```bash
python scripts/text_image_chat.py --help
python -m llava.eval.run_llava --help
```

모델은 첫 실행 시 Hugging Face에서 다운로드될 수 있으므로 네트워크와 디스크 공간을 미리 확인해 두는 것이 좋습니다.
