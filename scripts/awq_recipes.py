from compressed_tensors.quantization import QuantizationScheme
from llmcompressor.modifiers.awq import AWQMapping, AWQModifier


def build_qwen3_vl_awq_recipe() -> AWQModifier:
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
        # config_groups={
        #     "group_0": {
        #         "targets": ["Linear"],
        #         "weights": {
        #             "num_bits": 4,
        #             "type": "int",
        #             "symmetric": True,
        #             "group_size": 32,
        #             "strategy": "group",
        #             "dynamic": False,
        #             "actorder": None,
        #             "observer": "mse",
        #         },
        #     }
        # },
        config_groups=dict(
            group_0=QuantizationScheme.model_validate(
                {
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
            )
        ),
    )


_RECIPE_BUILDERS = {
    "qwen/qwen3-vl": build_qwen3_vl_awq_recipe,
}


def build_awq_recipe_for_model(model_id: str) -> AWQModifier:
    normalized = model_id.lower()
    for prefix, builder in _RECIPE_BUILDERS.items():
        if normalized.startswith(prefix):
            return builder()
    available = ", ".join(sorted(_RECIPE_BUILDERS.keys()))
    raise ValueError(
        f"No AWQ recipe defined for model_id '{model_id}'. Available prefixes: {available}"
    )
