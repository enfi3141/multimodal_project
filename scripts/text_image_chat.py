import argparse
import re
import sys
from io import BytesIO

import requests
from PIL import Image


def infer_conv_mode(model_name):
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    if "mistral" in name:
        return "mistral_instruct"
    if "v1.6-34b" in name:
        return "chatml_direct"
    if "v1" in name:
        return "llava_v1"
    if "mpt" in name:
        return "mpt"
    return "llava_v0"


def load_image(image_file):
    if image_file.startswith(("http://", "https://")):
        response = requests.get(image_file)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def read_prompt(prompt):
    if prompt is not None:
        return prompt

    if not sys.stdin.isatty():
        stdin_prompt = sys.stdin.read().strip()
        if stdin_prompt:
            return stdin_prompt

    return input("User: ").strip()


def add_image_token(
    prompt,
    model_config,
    has_image,
    default_image_token,
    image_placeholder,
    default_im_start_token,
    default_im_end_token,
):
    image_token = default_image_token
    if has_image and getattr(model_config, "mm_use_im_start_end", False):
        image_token = default_im_start_token + default_image_token + default_im_end_token

    if not has_image:
        if image_placeholder in prompt or default_image_token in prompt:
            raise ValueError("Prompt contains an image token but --image-file was not provided.")
        return prompt

    if image_placeholder in prompt:
        return re.sub(image_placeholder, image_token, prompt)

    if default_image_token in prompt:
        return prompt.replace(default_image_token, image_token, 1)

    return image_token + "\n" + prompt


def prepare_image_inputs(image_file, image_processor, model):
    import torch

    from llava.mm_utils import process_images

    if image_processor is None:
        raise ValueError("Image input requires a LLaVA multimodal checkpoint.")

    image = load_image(image_file)
    image_tensor = process_images([image], image_processor, model.config)
    image_dtype = torch.float32 if model.device.type == "cpu" else torch.float16

    if isinstance(image_tensor, list):
        image_tensor = [x.to(model.device, dtype=image_dtype) for x in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=image_dtype)

    return image_tensor, [image.size]


def generate_response(args):
    import torch

    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_PLACEHOLDER,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
        use_flash_attn=args.use_flash_attn,
    )

    conv_mode = infer_conv_mode(model_name)
    if args.conv_mode is not None:
        if args.conv_mode != conv_mode:
            print(
                "[WARNING] auto inferred conversation mode is {}, but using --conv-mode {}.".format(
                    conv_mode, args.conv_mode
                ),
                file=sys.stderr,
            )
        conv_mode = args.conv_mode

    user_prompt = read_prompt(args.prompt)
    if not user_prompt:
        raise ValueError("Prompt is empty.")

    image_tensor = None
    image_sizes = None
    if args.image_file is not None:
        image_tensor, image_sizes = prepare_image_inputs(args.image_file, image_processor, model)

    user_prompt = add_image_token(
        user_prompt,
        model.config,
        args.image_file is not None,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_PLACEHOLDER,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    generate_kwargs = {
        "do_sample": args.temperature > 0,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        generate_kwargs["top_p"] = args.top_p
    if image_tensor is not None:
        generate_kwargs["images"] = image_tensor
        generate_kwargs["image_sizes"] = image_sizes

    if args.debug:
        print({"model_name": model_name, "conv_mode": conv_mode, "prompt": prompt}, file=sys.stderr)

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **generate_kwargs)

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(output)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a LLaVA response from text, optionally grounded on one image."
    )
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt. If omitted, stdin/input() is used.")
    parser.add_argument("--image-file", type=str, default=None, help="Optional local image path or image URL.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    generate_response(build_parser().parse_args())
