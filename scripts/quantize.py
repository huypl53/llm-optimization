import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQMapping, AWQModifier
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
# Using a small subset of neuralmagic/calibration for speed
DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 64
MAX_SEQUENCE_LENGTH = 4096

print("Loading dataset...")
ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess_function(example):
    messages = []
    for message in example["messages"]:
        messages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
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


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    # Filter out pixel_values if they are missing (text-only calibration)
    res = {}
    for key, value in batch[0].items():
        if key == "pixel_values":
            res[key] = torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        else:
            res[key] = torch.tensor(value)
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
