import pytest
import torch

from dendritic.experiments.utils.experiment_finetuning import (
    FinetuningConfig,
    calculate_dendritic_trainable_params,
    calculate_matching_lora_rank,
    get_gpt2_layer_dimensions,
    setup_dendritic_model,
    setup_lora_model,
    get_trainable_params,
    PEFT_AVAILABLE,
)
from dendritic.experiments.utils.param_utils import count_parameters


@pytest.mark.parametrize("device", ["cpu"])
def test_dendritic_param_count_matches_calculation(device):
    config = FinetuningConfig()
    expected = calculate_dendritic_trainable_params(config)
    model = setup_dendritic_model(config, device)
    actual = get_trainable_params(model)
    tolerance = 0.01
    rel_diff = abs(actual - expected) / max(actual, expected)
    print(
        f"Dendritic - Expected: {expected}, Actual: {actual}, Rel diff: {rel_diff:.4f}"
    )
    assert rel_diff < tolerance, (
        f"Trainable param count mismatch: expected {expected}, got {actual}"
    )


@pytest.mark.skipif(not PEFT_AVAILABLE, reason="peft not installed")
@pytest.mark.parametrize("device", ["cpu"])
def test_lora_param_count_matches_calculation(device):
    # Create config with unique target modules (remove duplicate 'c_proj')
    config = FinetuningConfig()
    config.lora_target_modules = list(
        set(config.lora_target_modules)
    )  # Remove duplicates

    # Calculate expected dendritic params to determine matching LoRA rank
    dendritic_params = calculate_dendritic_trainable_params(config)
    lora_rank = calculate_matching_lora_rank(dendritic_params, config)

    # Set up LoRA model with calculated rank
    model = setup_lora_model(config, lora_rank, device)
    actual = get_trainable_params(model)

    # Calculate expected LoRA params
    layer_dims = get_gpt2_layer_dimensions()
    total_dim_sum = 0
    for module_name in config.lora_target_modules:
        if module_name in layer_dims:
            for in_dim, out_dim in layer_dims[module_name]:
                total_dim_sum += in_dim + out_dim

        # Special case: 'c_proj' targets both attn and mlp in GPT-2
        if module_name == "c_proj" and "c_proj_mlp" in layer_dims:
            for in_dim, out_dim in layer_dims["c_proj_mlp"]:
                total_dim_sum += in_dim + out_dim

    # Account for LoRA alpha parameters (one per target module)
    num_target_modules = len(set(config.lora_target_modules))
    alpha_params = num_target_modules * 12  # One alpha per module per layer (12 layers)

    # Debug prints
    print(f"\nDebug Info:")
    print(f"Target modules: {config.lora_target_modules}")
    print(f"Layer dimensions: {layer_dims}")
    print(f"Total dim sum: {total_dim_sum}")
    print(f"Calculated LoRA rank: {lora_rank}")
    print(f"Number of target modules: {num_target_modules}")
    print(f"Alpha parameters: {alpha_params}")
    print(f"Expected base params: {lora_rank * total_dim_sum}")
    print(f"Expected with alpha: {lora_rank * total_dim_sum + alpha_params}")
    print(f"Actual params: {actual}")

    # Print detailed parameter breakdown

    param_breakdown = count_parameters(model)
    print("\nParameter breakdown:")
    print(param_breakdown)

    # Update expected to include alpha parameters
    expected = lora_rank * total_dim_sum + alpha_params
    tolerance = 0.01
    print(f"\nLoRA - Expected: {expected}, Actual: {actual}")
    assert abs(actual - expected) / max(actual, expected) < tolerance, (
        f"LoRA trainable param count mismatch: expected {expected}, got {actual}"
    )
