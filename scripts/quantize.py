import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from llmcompressor import oneshot
from transformers import AutoModelForImageTextToText, AutoProcessor

from .awq_recipes import build_recipe_for_model
from .quantize_datasets import get_preprocess_fn


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize Qwen3-VL with AWQ.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the quantization config YAML.",
    )
    return parser.parse_args()


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


def load_quantize_config(path: str) -> QuantizeConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Quantization config must be a YAML mapping.")

    def require(key: str, expected_type: type) -> Any:
        if key not in data:
            raise ValueError(f"Missing required config field: {key}")
        value = data[key]
        if not isinstance(value, expected_type):
            raise ValueError(f"Config field '{key}' must be {expected_type.__name__}.")
        return value

    model_id = require("model_id", str)
    dataset_id = require("dataset_id", str)
    dataset_split = require("dataset_split", str)
    num_calibration_samples = require("num_calibration_samples", int)
    max_sequence_length = require("max_sequence_length", int)
    save_dir = data.get("save_dir") or f"{model_id.split('/')[-1]}-AWQ-INT4-512"
    if not isinstance(save_dir, str):
        raise ValueError("Config field 'save_dir' must be str if provided.")

    dataset_name = data.get("dataset_name")
    if dataset_name is not None and not isinstance(dataset_name, str):
        raise ValueError("Config field 'dataset_name' must be str or null.")

    seed = data.get("seed", 42)
    if not isinstance(seed, int):
        raise ValueError("Config field 'seed' must be int if provided.")

    return QuantizeConfig(
        model_id=model_id,
        save_dir=save_dir,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        num_calibration_samples=num_calibration_samples,
        max_sequence_length=max_sequence_length,
        seed=seed,
    )


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
    preprocess_fn = get_preprocess_fn(
        cfg.dataset_id, processor, cfg.max_sequence_length
    )
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
    args = parse_args()
    cfg = load_quantize_config(args.config)
    model, processor = load_model_and_processor(cfg)
    dataset = load_calibration_dataset(cfg)
    dataset = prepare_dataset(dataset, processor, cfg)
    recipe = build_recipe_for_model(cfg.model_id)
    run_quantization(model, processor, recipe, dataset, cfg)
    save_and_postprocess(model, processor, cfg.save_dir)


if __name__ == "__main__":
    main()
