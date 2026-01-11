import base64
import io
from typing import Any, Mapping

import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQMapping, AWQModifier
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# print('hello')
# exit()
# 1. SETUP
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"  # Ensure this matches your model
SAVE_DIR = MODEL_ID.split("/")[-1] + "-AWQ-INT4-512-norm"

print(f"Loading model {MODEL_ID}...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# 2. CALIBRATION DATASET
# Text-only (messages) or image-text (question+image+answer) are supported.
DATASET_ID = "Lin-Chen/MMStar"
DATASET_NAME = None  # e.g. "LLM" for neuralmagic/calibration
DATASET_SPLIT = "val"  # MMStar has a single "val" split
NUM_CALIBRATION_SAMPLES = 64
MAX_SEQUENCE_LENGTH = 4096

print("Loading dataset...")
split = f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"
if DATASET_NAME:
    ds = load_dataset(DATASET_ID, name=DATASET_NAME, split=split)
else:
    ds = load_dataset(DATASET_ID, split=split)
ds = ds.shuffle(seed=42)


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


def preprocess_function(example):
    if "messages" in example:
        messages = []
        for message in example["messages"]:
            messages.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                }
            )
    elif "image" in example and "question" in example:
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
    else:
        raise ValueError("Unsupported dataset schema for calibration.")

    inputs = processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == 1:
            inputs[key] = value.squeeze(0)

    return inputs


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    # Handle both text-only and image-text inputs; keep pixel values in bf16.
    res = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            stacked = torch.stack(values)
        else:
            stacked = torch.tensor(values)
        if key == "pixel_values":
            stacked = stacked.to(torch.bfloat16)
        res[key] = stacked
    return res


# 3. QUANTIZATION RECIPE
# FIX: : Explicitly list the Linear layers to avoid Hook errors.
# We also SKIP the vision tower ("visual") as it is sensitive to quantization.
recipe = AWQModifier(
    mappings=[
        AWQMapping(
            "re:.*input_layernorm$",
            ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
        ),
        # AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]), # feature size mis matched
        AWQMapping(
            "re:.*post_attention_layernorm$",
            ["re:.*gate_proj$", "re:.*up_proj$"],
        ),
        AWQMapping(
            "re:.*up_proj$",
            ["re:.*down_proj$"],
        ),
    ],
    ignore=[
        "re:.*embed_tokens",
        # "re:.*input_layernorm$",
        "re:.*mlp[.]gate$",
        # "re:.*post_attention_layernorm$",
        # "re:.*norm$",
        "re:model[.]visual.*",
        "re:visual.*",
        "re:.*visual.*",
        "lm_head",
    ],
    duo_scaling=True,
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "group_size": 32,
                "strategy": "group",
                "dynamic": False,
                "actorder": None,
                "observer": "mse",
            },
        }
    },
)


# 4. RUN QUANTIZATION
print("Starting OneShot Quantization...")
oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)

# 5. SAVE
print(f"Saving model to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
print("Done!")
