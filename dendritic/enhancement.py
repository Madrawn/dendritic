import os
import re
import torch
import torch.nn as nn
import datetime
from typing import Optional, List, Union, Dict, Any

from .layers.DendriticLayer import DendriticLayer


def enhance_model_with_dendritic(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    poly_rank: Union[int, str] = "auto",
    init_scale: float = 1e-6,
    freeze_linear: bool = True,
    verbose: bool = True,
    dendritic_cls: Optional[type] = None,
    dendritic_kwargs: Optional[dict] = None,
    **kwargs,
) -> nn.Module:
    """
    Enhance a pretrained model by replacing linear layers with dendritic versions.

    This allows retrofitting ANY pretrained neural network with polynomial feature
    interactions while maintaining exact initial behavior (identity initialization).

    Handles both nn.Linear and transformers.pytorch_utils.Conv1D layers.

    Args:
        model: PyTorch model to enhance
        target_layers: List of layer name patterns to replace (e.g., ['mlp.c_fc', 'attn.c_proj'])
                      If None, replaces all linear/Conv1D layers
        poly_rank: 'auto' for input_dim//16, or integer for fixed rank
        init_scale: Initial scale for polynomial pathway (should be ~1e-6 for identity)
        freeze_linear: Whether to freeze the pretrained linear weights
        verbose: Print conversion statistics
        dendritic_cls: Class to use for replacement (default: DendriticLayer; can be DendriticStack)
        dendritic_kwargs: Extra keyword arguments for dendritic_cls construction

    Returns:
        Enhanced model with dendritic layers

    Example:
        >>> from transformers import GPT2LMHeadModel
        >>> from dendritic.layer import DendriticLayer, DendriticStack
        >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
        >>> model = enhance_model_with_dendritic(
        ...     model,
        ...     target_layers=['mlp.c_fc'],
        ...     freeze_linear=True,
        ...     dendritic_cls=DendriticStack,
        ...     dendritic_kwargs={"bottleneck_dim": 128, "dropout": 0.1}
        ... )
        >>> # Model behavior is identical until you finetune
        >>> # Only ~0.05% new parameters are trainable
    """
    if poly_rank != "auto" and (not isinstance(poly_rank, int) or poly_rank <= 0):
        raise ValueError("poly_rank must be 'auto' or a positive integer")

    # Import default class if not provided
    if dendritic_cls is None:
        from .layers.DendriticLayer import DendriticLayer

        dendritic_cls = DendriticLayer
    if dendritic_kwargs is None:
        dendritic_kwargs = {}

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Get device of model
    device = next(model.parameters()).device

    # Check if Conv1D exists (HuggingFace models)
    try:
        from transformers.pytorch_utils import Conv1D

        has_conv1d = True
    except ImportError:
        Conv1D = None
        has_conv1d = False

    conversions = []
    replacements = []  # Store (parent, child_name, new_module) tuples
    dendritic_param_ids = set()

    # First pass: identify what needs to be replaced
    for name, module in model.named_modules():
        # Check if it's a Linear or Conv1D layer
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = has_conv1d and isinstance(module, Conv1D)

        if is_linear or is_conv1d:
            # Check if this layer should be converted
            should_convert = False
            if target_layers is None:
                should_convert = True
            else:
                # Filter out non-string patterns
                valid_patterns = [p for p in target_layers if isinstance(p, str)]
                should_convert = any(pattern in name for pattern in valid_patterns)

            if should_convert:
                # Get input/output dimensions
                if is_conv1d:
                    # Conv1D stores weights as (out_features, in_features) - transposed!
                    # But nf (num features) is output_dim, nx is input_dim
                    input_dim = module.weight.shape[0]  # This is actually in_features
                    output_dim = module.weight.shape[1]  # This is actually out_features
                    has_bias = module.bias is not None
                else:
                    # nn.Linear: (out_features, in_features)
                    input_dim = module.in_features
                    output_dim = module.out_features
                    has_bias = module.bias is not None

                # Calculate poly_rank
                if poly_rank == "auto":
                    pr = max(4, input_dim // 64)
                else:
                    pr = poly_rank

                # Build kwargs for construction
                layer_kwargs = dict(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    poly_rank=pr,
                    init_scale=init_scale,
                    bias=has_bias,
                )
                layer_kwargs.update(dendritic_kwargs)

                # Create dendritic layer ON THE SAME DEVICE
                dendritic = dendritic_cls(**layer_kwargs).to(device)

                # Custom patching logic for DendriticStack
                with torch.no_grad():
                    if (
                        hasattr(dendritic, "base_linear")
                        and dendritic_cls.__name__ == "DendriticStack"
                    ):
                        src_weight = (
                            module.weight
                            if not is_conv1d
                            else module.weight.t().contiguous()
                        )

                        # 1. Copy original weights to base_linear (The Identity Path)
                        dendritic.base_linear.weight.copy_(src_weight)
                        if has_bias and dendritic.base_linear.bias is not None:
                            dendritic.base_linear.bias.copy_(module.bias)

                        # 2. ZERO OUT the new branch (The Stack Path)
                        if hasattr(dendritic.layer2, "linear"):
                            dendritic.layer2.linear.weight.zero_()
                            if dendritic.layer2.linear.bias is not None:
                                dendritic.layer2.linear.bias.zero_()

                            # Zero out polynomial components
                            dendritic.layer1.scale.fill_(0.0)
                            dendritic.layer2.scale.fill_(0.0)
                            if hasattr(dendritic.layer1, "diag_scale"):
                                dendritic.layer1.diag_scale.fill_(0.0)
                            if hasattr(dendritic.layer2, "diag_scale"):
                                dendritic.layer2.diag_scale.fill_(0.0)

                    else:
                        # For DendriticLayer
                        if hasattr(dendritic, "linear") and hasattr(module, "weight"):
                            if is_conv1d:
                                # For Conv1D layers, transpose weight to match Linear format
                                src_weight = module.weight.t().contiguous()
                            else:
                                src_weight = module.weight

                            # Copy original weights to linear pathway
                            dendritic.linear.weight.copy_(src_weight)
                            if has_bias:
                                dendritic.linear.bias.copy_(module.bias)
                        if hasattr(dendritic, "scale"):
                            dendritic.scale.fill_(init_scale)
                        if hasattr(dendritic, "diag_scale"):
                            dendritic.diag_scale.fill_(init_scale)

                # Freeze linear pathway
                if freeze_linear:
                    if hasattr(dendritic, "linear"):
                        dendritic.linear.weight.requires_grad = False
                        if dendritic.linear.bias is not None:
                            dendritic.linear.bias.requires_grad = False

                    if hasattr(dendritic, "base_linear"):
                        dendritic.base_linear.weight.requires_grad = False
                        if dendritic.base_linear.bias is not None:
                            dendritic.base_linear.bias.requires_grad = False

                # Collect dendritic parameters
                for n, p in dendritic.named_parameters():
                    # Mark all dendritic pathway parameters as trainable
                    if (
                        n.startswith("w1")
                        or n.startswith("w2")
                        or n.startswith("w_")
                        or n.startswith("scale")
                        or n.startswith("diag")
                        or n.startswith("poly_")
                        or n.startswith("w_diag")  # Include diagonal pathway weights
                    ):
                        p.requires_grad = True
                        dendritic_param_ids.add(id(p))

                # Track statistics
                linear_params = sum(p.numel() for p in module.parameters())
                dendritic_params = sum(p.numel() for p in dendritic.parameters())
                trainable_params = sum(
                    p.numel() for p in dendritic.parameters() if p.requires_grad
                )

                conversions.append(
                    {
                        "name": name,
                        "linear_params": linear_params,
                        "dendritic_params": dendritic_params,
                        "added_params": dendritic_params - linear_params,
                        "trainable_params": trainable_params,
                        "poly_rank": pr,
                    }
                )

                # Store replacement info
                if "." in name:
                    parent_name = name.rsplit(".", 1)[0]
                    child_name = name.split(".")[-1]
                    parent = model.get_submodule(parent_name)
                else:
                    # Direct child of root module
                    parent = model
                    child_name = name

                replacements.append((parent, child_name, dendritic))
    if not replacements:
        raise TypeError(
            "Warning: No layers were converted. Check target_layers patterns."
        )
    # Second pass: actually replace modules
    for parent, child_name, new_module in replacements:
        setattr(parent, child_name, new_module)

    # Print trainable parameter names for verification
    if verbose:
        print("\nTrainable parameters after enhancement:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"  {n} | shape={tuple(p.shape)}")

        if conversions:
            total_params_before = sum(c["linear_params"] for c in conversions)
            total_params_after = sum(c["dendritic_params"] for c in conversions)
            total_trainable = sum(c["trainable_params"] for c in conversions)

            print(f"\n{'='*70}")
            print("Dendritic Enhancement Summary")
            print(f"{'='*70}")
            print(f"Converted {len(conversions)} layer(s)")
            print(f"Parameters before:  {total_params_before:>12,}")
            print(f"Parameters after:   {total_params_after:>12,}")
            print(
                f"Added parameters:   {total_params_after - total_params_before:>12,} "
                f"(+{(total_params_after/total_params_before-1)*100:.2f}%)"
            )
            print(
                f"\nTrainable params:   {total_trainable:>12,} "
                f"({100*total_trainable/total_params_after:.2f}% of enhanced layers)"
            )
    model.dendritic_config = {  # type: ignore
        "target_layers": target_layers,
        "poly_rank": poly_rank,
        "init_scale": init_scale,
        "freeze_linear": freeze_linear,
        "verbose": verbose,
        "dendritic_cls": dendritic_cls,
        "dendritic_kwargs": dendritic_kwargs,
        **kwargs,
    }
    return model


def verify_identity_initialization(model_original, model_enhanced, test_input):
    """
    Verify that the enhanced model produces identical outputs to the original.

    Args:
        model_original: Original pretrained model
        model_enhanced: Enhanced model with dendritic layers
        test_input: Sample input tensor

    Returns:
        max_diff: Maximum absolute difference in outputs
    """
    model_original.eval()
    model_enhanced.eval()

    with torch.no_grad():
        out_original = model_original(test_input)
        out_enhanced = model_enhanced(test_input)

        # Handle different output types
        if isinstance(out_original, torch.Tensor):
            diff = (out_original - out_enhanced).abs().max().item()
        else:
            # For model outputs with .logits
            diff = (out_original.logits - out_enhanced.logits).abs().max().item()

    return diff


def get_polynomial_stats(model):
    """
    Get statistics about polynomial pathways in the model.

    Returns dict with scale values for each dendritic layer.
    """
    from .layers.DendriticStack import DendriticStack

    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (DendriticLayer, DendriticStack)):
            with torch.no_grad():
                scale = getattr(module, "scale", 0.0)
                # Only call .item() if scale is a tensor
                if isinstance(scale, torch.Tensor):
                    scale = scale.item()

                # Handle DendriticStack layers
                if hasattr(module, "layer1") and hasattr(module.layer1, "scale"):
                    layer1_scale = module.layer1.scale
                    if isinstance(layer1_scale, torch.Tensor):
                        layer1_scale = layer1_scale.item()
                    scale = (scale + layer1_scale) / 2
                if hasattr(module, "layer2") and hasattr(module.layer2, "scale"):
                    layer2_scale = module.layer2.scale
                    if isinstance(layer2_scale, torch.Tensor):
                        layer2_scale = layer2_scale.item()
                    scale = (scale + layer2_scale) / 2

                # Effective rank calculation
                try:
                    if hasattr(module, "w1"):
                        _, S, _ = torch.svd(module.w1)
                        eff_rank = (S.sum() ** 2) / (S**2).sum()
                        eff_rank = (
                            eff_rank.item()
                            if isinstance(eff_rank, torch.Tensor)
                            else eff_rank
                        )
                    else:
                        eff_rank = None
                except:
                    eff_rank = None

                stats[name] = {
                    "scale": abs(scale),
                    "eff_rank": eff_rank,
                    "poly_rank": getattr(module, "poly_rank", "N/A"),
                }

    return stats


def extract_dendritic_state(model: nn.Module) -> Dict[str, Any]:
    """
    Extracts trainable parameters AND the configuration required to reconstruct
    the dendritic layers.
    
    Returns a dictionary payload containing:
    - 'dendritic_config': The args needed to re-enhance the base model.
    - 'state_dict': The specific weights for the dendritic layers.
    """
    if not hasattr(model, "dendritic_config"):
        raise AttributeError(
            "Model is missing 'dendritic_config'. "
            "Ensure the model was enhanced using 'enhance_model_with_dendritic'."
        )

    # 1. Identify keys that are currently trainable (the dendritic weights)
    # We use a set for O(1) lookups during the state_dict filter
    trainable_keys = {name for name, param in model.named_parameters() if param.requires_grad}

    # 2. Get the full standard state_dict (handles buffers and canonical naming automatically)
    full_state = model.state_dict()

    # 3. Filter: Keep only the trainable keys
    # Note: If your dendritic layers have buffers (non-trainable state), 
    # you might want to modify this logic to include those too.
    dendritic_weights = {k: v.cpu().clone() for k, v in full_state.items() if k in trainable_keys}

    return {
        "dendritic_config": model.dendritic_config,
        "state_dict": dendritic_weights,
        "_metadata": {
            "version": "2.0",
            "type": "dendritic_bundle"
        }
    }


def apply_dendritic_state(base_model: nn.Module, state_payload: Dict[str, Any]) -> nn.Module:
    """
    Takes a clean BASE model and a dendritic state payload.
    1. Reads the config from the payload.
    2. Enhances the base model (architecture change).
    3. Loads the weights.
    
    Returns the Enhanced Model.
    """
    config = state_payload.get("dendritic_config")
    weights = state_payload.get("state_dict")

    if not config or weights is None:
        raise ValueError("Invalid dendritic state payload: missing config or weights.")

    # 1. Enhance the model (Modify Architecture)
    # We assume 'enhance_model_with_dendritic' is available in your scope
    enhanced_model = enhance_model_with_dendritic(base_model, **config)

    # 2. Load the weights
    # strict=False allows the base model weights (which are missing from the payload)
    # to remain as they are (initialized/frozen).
    missing, unexpected = enhanced_model.load_state_dict(weights, strict=False)

    # 3. Validation
    # We expect missing keys (the frozen base weights). 
    # We DO NOT expect unexpected keys (weights in file that don't match config).
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys found while loading dendritic state.")
    
    return enhanced_model


def save_dendritic_model(model: nn.Module, filepath: str) -> None:
    """Wrapper to save the extracted state to disk."""
    payload = extract_dendritic_state(model)
    
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
    torch.save(payload, filepath)


def load_dendritic_model(base_model: nn.Module, filepath: str) -> nn.Module:
    """Wrapper to load from disk and apply to a base model."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model file found at {filepath}")
        
    payload = torch.load(filepath, map_location="cpu")
    return apply_dendritic_state(base_model, payload)


# Example usage
if __name__ == "__main__":
    print("Example: Enhancing and saving model state\n")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    # Create and enhance model
    model = SimpleModel()
    enhanced = enhance_model_with_dendritic(
        model, target_layers=["fc1"], poly_rank=16, freeze_linear=True
    )

    # Extract and save dendritic state
    state = extract_dendritic_state(enhanced)
    print(f"Extracted state with {state['_metadata']['num_params']} parameters")
    print(
        "Parameter keys:", [k for k in state.keys() if not k.startswith("_")][:5], "..."
    )

    # Create new enhanced model with saved state
    new_enhanced = create_dendritic_state(
        model,
        state_dict=state,
        enhancement_params={
            "target_layers": ["fc1"],
            "poly_rank": 16,
            "freeze_linear": True,
        },
    )

    # Verify same outputs
    test_input = torch.randn(2, 128)
    diff = verify_identity_initialization(enhanced, new_enhanced, test_input)
    print(f"Model output difference after loading state: {diff:.6f}")
