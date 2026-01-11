import argparse
import asyncio
import csv
import time
from collections import defaultdict
from itertools import cycle, islice
from pathlib import Path

from PIL import Image
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


def iter_image_paths(image_dir: Path) -> list[Path]:
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed
    )


def maybe_augment_image(image_pil: Image.Image, occurrence_idx: int) -> Image.Image:
    if occurrence_idx <= 0:
        return image_pil
    img = image_pil.copy()
    x = occurrence_idx % img.width
    y = (occurrence_idx * 7) % img.height
    r, g, b = img.getpixel((x, y))
    img.putpixel((x, y), ((r + occurrence_idx) % 256, (g + 1) % 256, (b + 1) % 256))
    return img


def build_prompt_text(prompt: str, system_prompt: str, mm_placeholder: str) -> str:
    user_block = (
        prompt
        if mm_placeholder in prompt
        else f"{mm_placeholder}\n{prompt}"
    )
    if system_prompt:
        return f"{system_prompt}\nUSER: {user_block}\nASSISTANT:"
    return f"USER: {user_block}\nASSISTANT:"


def build_chat_prompt(
    image_pil: Image.Image, prompt: str, system_prompt: str
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image_pil},
                {"type": "text", "text": prompt},
            ],
        },
    ]


async def generate_one(
    engine: AsyncLLM, prompt: dict, sampling_params: SamplingParams, request_id: str
) -> str:
    final_output = None
    async for output in engine.generate(
        request_id=request_id, prompt=prompt, sampling_params=sampling_params
    ):
        final_output = output
        if output.finished:
            break
    if not final_output or not final_output.outputs:
        return ""
    return final_output.outputs[0].text


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run async vLLM inference on each image in a local directory."
    )
    parser.add_argument("image_dir", type=Path, help="Directory with images to load.")
    parser.add_argument(
        "--model",
        default="llava-hf/llava-1.5-7b-hf",
        help="vLLM model id.",
    )
    parser.add_argument(
        "--prompt",
        default="What's in this image?",
        help="Text prompt to pair with each image.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a text file to use as the prompt (overrides --prompt).",
    )
    parser.add_argument(
        "--system-prompt",
        default="Bạn là chuyên gia phân tích hình ảnh",
        help="System prompt for the conversation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs.csv"),
        help="Path to write CSV rows of image name, response, and latency.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
        help="Fraction of GPU memory to use for vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Maximum model sequence length.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to process from the directory.",
    )
    parser.add_argument(
        "--mm-placeholder",
        default="<image>",
        help="Image placeholder token required by the model's prompt format.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("raw", "chat"),
        default="raw",
        help="Use raw prompt text with multimodal placeholder. Chat is unsupported.",
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens to generate.", default=256
    )
    parser.add_argument("--min-tokens", type=int, help="Minimum tokens to generate.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, help="Top-p nucleus sampling.")
    parser.add_argument("--top-k", type=int, help="Top-k sampling.")
    args = parser.parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {args.image_dir}")

    if args.prompt_file:
        if not args.prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        if not args.prompt_file.is_file():
            raise FileNotFoundError(f"Prompt file is not a file: {args.prompt_file}")
        prompt_text = args.prompt_file.read_text(encoding="utf-8").rstrip("\n")
    else:
        prompt_text = args.prompt

    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be a positive integer")

    image_paths = iter_image_paths(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"No supported images found in {args.image_dir}")
    if args.max_images is not None:
        if args.max_images <= len(image_paths):
            image_paths = image_paths[: args.max_images]
        else:
            image_paths = list(islice(cycle(image_paths), args.max_images))

    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    sampling_params = SamplingParams()
    if args.max_tokens is not None:
        sampling_params.max_tokens = args.max_tokens
    if args.min_tokens is not None:
        sampling_params.min_tokens = args.min_tokens
    if args.temperature is not None:
        sampling_params.temperature = args.temperature
    if args.top_p is not None:
        sampling_params.top_p = args.top_p
    if args.top_k is not None:
        sampling_params.top_k = args.top_k

    image_cache: dict[Path, Image.Image] = {}
    seen_counts = defaultdict(int)
    latencies = []

    try:
        with args.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "response", "latency_s"])

            for idx, image_path in enumerate(image_paths, start=1):
                if image_path not in image_cache:
                    with Image.open(image_path) as img:
                        image_cache[image_path] = img.convert("RGB")
                base_image = image_cache[image_path]

                occurrence_idx = seen_counts[image_path]
                seen_counts[image_path] += 1
                image_pil = maybe_augment_image(base_image, occurrence_idx)

                if args.prompt_format == "chat":
                    raise ValueError(
                        "AsyncLLM.generate does not accept chat message lists. "
                        "Use --prompt-format raw and set --mm-placeholder."
                    )
                full_prompt = build_prompt_text(
                    prompt_text, args.system_prompt, args.mm_placeholder
                )
                prompt = {
                    "prompt": full_prompt,
                    "multi_modal_data": {"image": image_pil},
                }

                start = time.perf_counter()
                generated_text = await generate_one(
                    engine=engine,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=f"image-{idx}",
                )
                latency = time.perf_counter() - start
                latencies.append(latency)

                writer.writerow([image_path.name, generated_text, f"{latency:.4f}"])
                suffix = (
                    f" [resample {occurrence_idx + 1}]" if occurrence_idx > 0 else ""
                )
                print(
                    f"{image_path.name}{suffix}: {generated_text} (latency {latency:.4f}s)"
                )

            avg_latency = sum(latencies) / len(latencies)
            writer.writerow(
                ["SUMMARY", f"tested={len(latencies)}", f"{avg_latency:.4f}"]
            )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
