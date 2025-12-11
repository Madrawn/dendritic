import torch
import torch.nn as nn
import datetime
from typing import Optional, List, Union, Dict, Any


def enhance_model_with_dendritic(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    poly_rank: Union[int, str] = "auto",
    init_scale: float = 1e-6,
    freeze_linear: bool = True,
    verbose: bool = True,
    dendritic_cls: Optional[type] = None,
    dendritic_kwargs: Optional[dict] = None,
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
    # Import default class if not provided
    if dendritic_cls is None:
        from .layer import DendriticLayer

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
            should_convert = target_layers is None or any(
                pattern in name for pattern in target_layers
            )

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
                                dendritic.linear.weight.copy_(
                                    module.weight.t().contiguous()
                                )
                            else:
                                dendritic.linear.weight.copy_(module.weight)
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
                    if (
                        n.startswith("w1")
                        or n.startswith("w2")
                        or n.startswith("w_")
                        or n.startswith("scale")
                        or n.startswith("diag")
                        or n.startswith("poly_")
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
    from .layer import DendriticLayer, DendriticStack

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
    Extract only the trainable dendritic parameters from an enhanced model.

    This creates a much smaller state dict than saving the entire model,
    containing only the actual trained parameters (polynomial pathways etc.)
    while excluding the original frozen weights.

    Args:
        model: Enhanced model with dendritic layers

    Returns:
        Dictionary containing:
        - Trainable dendritic parameters
        - Metadata with version, timestamp, and parameter count

    Example:
        >>> enhanced = enhance_model_with_dendritic(model, target_layers=['mlp.c_fc'])
        >>> dendritic_state = extract_dendritic_state(enhanced)
        >>> torch.save(dendritic_state, 'dendritic_weights.pt')
    """
    state = {
        "_metadata": {
            "version": "1.0",
            "timestamp": datetime.datetime.now().isoformat(),
            "description": "Extracted dendritic state (trainable parameters only)",
        }
    }

    # Track parameters
    num_params = 0

    # Extract parameters from all modules
    for module in model.modules():
        # Handle DendriticLayers and custom components
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                state[f"{module._get_name()}.{name}"] = param.data.cpu().clone()
                num_params += 1

    # Update metadata
    state["_metadata"]["num_params"] = num_params

    return state


def load_dendritic_state(model: nn.Module, state_dict: Dict[str, Any]) -> nn.Module:
    """
    Load dendritic state dict into an already enhanced model.

    Args:
        model: Enhanced model (must be compatible with the state_dict)
        state_dict: State dictionary from extract_dendritic_state()

    Returns:
        Model with loaded dendritic state

    Note:
        Model must have the same architecture as the one used to create the state_dict.
        Only parameters that exist in the model and are trainable will be updated.

    Example:
        >>> enhanced = enhance_model_with_dendritic(model, target_layers=['mlp.c_fc'])
        >>> dendritic_state = torch.load('dendritic_weights.pt')
        >>> enhanced = load_dendritic_state(enhanced, dendritic_state)
    """
    # Remove metadata for loading
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_")}

    # Load parameters
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            if model_state[name].shape == param.shape:
                model_state[name].copy_(param)
            else:
                print(
                    f"Warning: Shape mismatch for {name} "
                    f"(expected {model_state[name].shape}, got {param.shape})"
                )

    model.load_state_dict(model_state, strict=False)
    return model


def create_dendritic_state(
    base_model: nn.Module,
    state_dict: Optional[Dict[str, Any]] = None,
    enhancement_params: Optional[Dict] = None,
) -> nn.Module:
    """
    Create or recreate an enhanced model with optional stored state.

    This is a helper function that combines model enhancement with state loading.
    It ensures the model is properly enhanced before loading any saved state.

    Args:
        base_model: Base pre-trained model to enhance
        state_dict: Optional state from extract_dendritic_state()
        enhancement_params: Parameters for enhance_model_with_dendritic

    Returns:
        Enhanced model with loaded state (if provided)

    Example:
        >>> base_model = GPT2LMHeadModel.from_pretrained('gpt2')
        >>> dendritic_state = torch.load('dendritic_weights.pt')
        >>> enhanced = create_dendritic_state(
        ...     base_model,
        ...     state_dict=dendritic_state,
        ...     enhancement_params={
        ...         'target_layers': ['mlp.c_fc'],
        ...         'poly_rank': 32,
        ...         'freeze_linear': True
        ...     }
        ... )
    """
    if enhancement_params is None:
        enhancement_params = {}

    # Create enhanced model
    enhanced_model = enhance_model_with_dendritic(base_model, **enhancement_params)

    # Load state if provided
    if state_dict is not None:
        enhanced_model = load_dendritic_state(enhanced_model, state_dict)

    return enhanced_model


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
