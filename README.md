# Quantization (AWQ)

This project runs AWQ quantization for Qwen3-VL using `llmcompressor` and a YAML-driven config.

## What changed

- Quantization reads a **single YAML config** for dataset/model settings.
- AWQ recipes are defined in Python and selected by model in `scripts/awq_recipes.py`.
- Dataset-specific preprocessing lives in a separate module.
- A sample config is provided under `configs/`.

## Requirements

- Python 3.12+
- Dependencies in `pyproject.toml`

## Quick start

Use the sample config:

```bash
python scripts/quantize.py --config configs/quantize.yaml
```

## Quantize config YAML

Required fields:
- `model_id` (str)
- `dataset_id` (str)
- `dataset_split` (str)
- `num_calibration_samples` (int)
- `max_sequence_length` (int)

Optional fields:
- `save_dir` (str)
- `dataset_name` (str | null)
- `seed` (int)

Example:

```yaml
model_id: Qwen/Qwen3-VL-4B-Instruct
save_dir: Qwen3-VL-4B-Instruct-AWQ-INT4-512
dataset_id: neuralmagic/calibration
dataset_name: LLM
dataset_split: train
num_calibration_samples: 64
max_sequence_length: 16384
seed: 42
```

## AWQ recipe (Python)

Recipes are defined in `scripts/awq_recipes.py` and picked based on `model_id`.
To add a new model recipe, implement a builder and register it in `_RECIPE_BUILDERS`.

## Dataset preprocessing

Dataset-specific preprocessing is implemented in:

- `scripts/quantize_datasets.py`

To add a new dataset:
1. Add a preprocessing function that converts a dataset example into model inputs.
2. Register it in `DATASET_PREPROCESSORS` by `dataset_id`.
3. Use that `dataset_id` in your quantize config YAML.
