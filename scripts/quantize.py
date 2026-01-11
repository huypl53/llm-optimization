import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQMapping, AWQModifier
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


@dataclass(frozen=True)
class QuantizeConfig:
    model_id: str
    save_dir: str
    dataset_id: str
    dataset_name: str | None
    dataset_split: str
    num_calibration_samples: int
    max_sequence_length: int
    seed: int = 42


def build_default_config() -> QuantizeConfig:
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    return QuantizeConfig(
        model_id=model_id,
        save_dir=f"{model_id.split('/')[-1]}-AWQ-INT4-512-norm",
        dataset_id="Lin-Chen/MMStar",
        dataset_name=None,
        dataset_split="val",
        num_calibration_samples=64,
        max_sequence_length=4096,
    )


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


def remove_keys_nested(obj: Any, keys: set[str]) -> bool:
    removed = False
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in keys:
                obj.pop(key, None)
                removed = True
                continue
            if remove_keys_nested(obj.get(key), keys):
                removed = True
    elif isinstance(obj, list):
        for item in obj:
            if remove_keys_nested(item, keys):
                removed = True
    return removed


def build_preprocess_fn(processor: AutoProcessor, max_sequence_length: int):
    def preprocess_function(example: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "messages" in example:
            messages = [
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                }
                for message in example["messages"]
            ]
        elif "image" in example and "question" in example:
            if example["image"] is None:
                raise ValueError(
                    "Image is required for image-text calibration samples."
                )
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

    return preprocess_function


def data_collator(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    res: dict[str, torch.Tensor] = {}
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


def build_recipe() -> AWQModifier:
    return AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm$",
                ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
            ),
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
            "re:.*mlp[.]gate$",
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


def load_model_and_processor(cfg: QuantizeConfig):
    print(f"Loading model {cfg.model_id}...")
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    return model, processor


def load_calibration_dataset(cfg: QuantizeConfig):
    print("Loading dataset...")
    split = f"{cfg.dataset_split}[:{cfg.num_calibration_samples}]"
    if cfg.dataset_name:
        ds = load_dataset(cfg.dataset_id, name=cfg.dataset_name, split=split)
    else:
        ds = load_dataset(cfg.dataset_id, split=split)
    return ds.shuffle(seed=cfg.seed)


def prepare_dataset(ds, processor: AutoProcessor, cfg: QuantizeConfig):
    preprocess_fn = build_preprocess_fn(processor, cfg.max_sequence_length)
    return ds.map(preprocess_fn, batched=False, remove_columns=ds.column_names)


def run_quantization(model, processor, recipe, dataset, cfg: QuantizeConfig):
    print("Starting OneShot Quantization...")
    oneshot(
        model=model,
        processor=processor,
        recipe=recipe,
        dataset=dataset,
        max_seq_length=cfg.max_sequence_length,
        num_calibration_samples=cfg.num_calibration_samples,
        data_collator=data_collator,
    )


def postprocess_config_json(save_dir: str) -> None:
    config_path = Path(save_dir) / "config.json"
    if not config_path.exists():
        return
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    removed = remove_keys_nested(config, {"scale_dtype", "zp_dtype"})
    if removed:
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)


def postprocess_tokenizer_json(save_dir: str) -> None:
    tokenizer_path = Path(save_dir) / "tokenizer.json"
    if not tokenizer_path.exists():
        return
    with tokenizer_path.open("r", encoding="utf-8") as handle:
        tokenizer_config = json.load(handle)
    if tokenizer_config.get("truncation") is not None:
        tokenizer_config["truncation"] = None
        with tokenizer_path.open("w", encoding="utf-8") as handle:
            json.dump(tokenizer_config, handle, indent=2)


def save_and_postprocess(model, processor, save_dir: str) -> None:
    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir, save_compressed=True)
    processor.save_pretrained(save_dir)
    postprocess_config_json(save_dir)
    postprocess_tokenizer_json(save_dir)
    print("Done!")


def main() -> None:
    cfg = build_default_config()
    model, processor = load_model_and_processor(cfg)
    dataset = load_calibration_dataset(cfg)
    dataset = prepare_dataset(dataset, processor, cfg)
    recipe = build_recipe()
    run_quantization(model, processor, recipe, dataset, cfg)
    save_and_postprocess(model, processor, cfg.save_dir)


if __name__ == "__main__":
    main()
