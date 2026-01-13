import base64
import io
from typing import Any, Callable, Mapping

import torch
from PIL import Image
from transformers import AutoProcessor


def convert_image_mode(image: Image.Image, mode: str) -> Image.Image:
    return image if image.mode == mode else image.convert(mode)


def process_image(image: Any) -> Mapping[str, Any]:
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(io.BytesIO(image["bytes"]))
    if isinstance(image, Image.Image):
        image = convert_image_mode(image, "RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
        }
    if isinstance(image, str):
        image_url = (
            image
            if image.startswith(("http://", "https://", "file://"))
            else f"file://{image}"
        )
        return {"type": "image_url", "image_url": {"url": image_url}}

    try:
        import numpy as np
    except Exception:
        np = None

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(image, (list, tuple)):
        if np is None:
            raise ValueError("numpy is required to process array images")
        image = np.array(image)
    if np is not None and isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            max_val = float(image.max()) if image.size else 0.0
            if max_val <= 1.0:
                image = (image * 255.0).clip(0, 255).astype("uint8")
            else:
                image = image.clip(0, 255).astype("uint8")
        image = Image.fromarray(image)
        return process_image(image)

    raise ValueError(
        "Invalid image input. Must be PIL.Image.Image, str, dict with bytes, or array."
    )


def apply_chat_template(
    processor: AutoProcessor, messages: list[dict[str, Any]], max_sequence_length: int
) -> dict[str, torch.Tensor]:
    inputs = processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_sequence_length,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == 1:
            inputs[key] = value.squeeze(0)

    return inputs


def preprocess_neuralmagic_calibration(
    example: dict[str, Any],
    processor: AutoProcessor,
    max_sequence_length: int,
) -> dict[str, torch.Tensor]:
    if "messages" not in example:
        raise ValueError("Expected 'messages' field for neuralmagic calibration.")
    messages = [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in example["messages"]
    ]
    return apply_chat_template(processor, messages, max_sequence_length)


def preprocess_mmstar(
    example: dict[str, Any],
    processor: AutoProcessor,
    max_sequence_length: int,
) -> dict[str, torch.Tensor]:
    if "image" not in example or "question" not in example:
        raise ValueError("Expected 'image' and 'question' fields for MMStar.")
    if example["image"] is None:
        raise ValueError("Image is required for image-text calibration samples.")
    user_content = [
        process_image(example["image"]),
        {"type": "text", "text": example["question"]},
    ]
    messages = [{"role": "user", "content": user_content}]
    if "answer" in example and example["answer"] is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(example["answer"])}],
            }
        )
    return apply_chat_template(processor, messages, max_sequence_length)


DATASET_PREPROCESSORS: dict[str, Callable[..., dict[str, torch.Tensor]]] = {
    "neuralmagic/calibration": preprocess_neuralmagic_calibration,
    "Lin-Chen/MMStar": preprocess_mmstar,
}


def get_preprocess_fn(
    dataset_id: str, processor: AutoProcessor, max_sequence_length: int
) -> Callable[[dict[str, Any]], dict[str, torch.Tensor]]:
    if dataset_id not in DATASET_PREPROCESSORS:
        available = ", ".join(sorted(DATASET_PREPROCESSORS))
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. Available: {available}"
        )
    handler = DATASET_PREPROCESSORS[dataset_id]

    def preprocess_function(example: dict[str, Any]) -> dict[str, torch.Tensor]:
        return handler(
            example,
            processor=processor,
            max_sequence_length=max_sequence_length,
        )

    return preprocess_function
