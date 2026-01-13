# Quantization (AWQ)

This project runs AWQ quantization for Qwen3-VL using `llmcompressor` and a YAML-driven configuration.

## What changed

- Quantization now reads **two YAML files**: a quantize config and an AWQ recipe.
- Dataset-specific preprocessing lives in a separate module.
- Sample YAMLs are provided under `configs/`.

## Requirements

- Python 3.12+
- Dependencies in `pyproject.toml`

## Quick start

Use the sample configs:

```bash
python scripts/quantize.py --config configs/quantize.yaml --recipe configs/recipe.yaml
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

## AWQ recipe YAML

The recipe is loaded from a YAML file. It accepts an `AWQModifier` root (recommended) or a raw mapping.

Example:

```yaml
AWQModifier:
  mappings:
    - smooth_layer: "re:.*self_attn_layer_norm"
      balance_layers: ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"]
    - smooth_layer: "re:.*final_layer_norm"
      balance_layers: ["re:.*fc1"]
  ignore: ["lm_head"]
  config_groups:
    group_0:
      targets:
        - "Linear"
      input_activations: null
      output_activations: null
      weights:
        num_bits: 4
        type: int
        symmetric: false
        strategy: group
        group_size: 128
```

## Dataset preprocessing

Dataset-specific preprocessing is implemented in:

- `scripts/quantize_datasets.py`

To add a new dataset:
1. Add a preprocessing function that converts a dataset example into model inputs.
2. Register it in `DATASET_PREPROCESSORS` by `dataset_id`.
3. Use that `dataset_id` in your quantize config YAML.
