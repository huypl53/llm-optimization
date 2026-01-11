from transformers import AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

print(f"Loading configuration for {MODEL_ID}...")
# Load with meta device to avoid using RAM/VRAM just for inspection
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, trust_remote_code=True, device_map="meta"
)

print("\n====== MODEL ARCHITECTURE (Truncated) ======")
print(
    "We will print the first layer of the language model to understand the naming structure.\n"
)

# Iterate through named modules
for name, module in model.named_modules():
    # Print the Visual encoder top-level (just to verify name)
    if "visual" in name and len(name.split(".")) < 4:
        print(f"TYPE: {type(module).__name__} | NAME: {name}")

    # Print ONLY the first text decoder layer (usually layer 0)
    # This is where we need to find our Linear targets.
    if "model.layers.0" in name:
        print(f"TYPE: {type(module).__name__} | NAME: {name}")
    if "model.visual.blocks.0" in name:
        print(f"TYPE: {type(module).__name__} | NAME: {name}")

import re

input_layernorms = [
    name for name, _ in model.named_modules() if re.match(r".*input_layernorm$", name)
]
print(f"input_layernorms: {input_layernorms}")
print("\n============================================")
print("Use the names ending in 'Linear' above as your regex targets.")
