import os
import logging
import torch
import torch.nn as nn
from typing import Literal, Optional, List, Union, Dict, Any, Tuple, Set, Type, cast
import torch
from torch import Tensor
from dataclasses import dataclass
from transformers.modeling_utils import PreTrainedModel

from .layers.DendriticLayer import DendriticLayer

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class LayerConversionStats:
    """Data class to store statistics about layer conversion."""
    name: str
    linear_params: int
    dendritic_params: int
    trainable_params: int
    poly_rank: int

    @property
    def added_params(self) -> int:
        return self.dendritic_params - self.linear_params

    @property
    def trainable_percentage(self) -> float:
        return (self.trainable_params / self.dendritic_params) * 100 if self.dendritic_params > 0 else 0.0


class DendriticEnhancementError(Exception):
    """Base exception for dendritic enhancement errors."""
    pass


class NoLayersConvertedError(DendriticEnhancementError):
    """Raised when no layers are converted during enhancement."""
    pass


def _get_conv1d_class() -> Optional[Type]:
    """Safely import and return Conv1D class if available."""
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except ImportError:
        return None


def _validate_enhancement_params(
    model: nn.Module,
    poly_rank: Union[int, Literal["auto"]],
    target_layers: Optional[List[str]]
) -> None:
    """Validate enhancement parameters."""
    if not isinstance(model, PreTrainedModel):
        raise TypeError("model must be a PyTorch PreTrainedModel")
    
    if poly_rank != "auto" and (not isinstance(poly_rank, int) or poly_rank <= 0):
        raise ValueError("poly_rank must be 'auto' or a positive integer")
    
    if target_layers is not None and not all(isinstance(p, str) for p in target_layers):
        raise TypeError("All target_layers patterns must be strings")
    
    if target_layers is not None and not all(isinstance(p, str) for p in target_layers):
        raise TypeError("All target_layers patterns must be strings")


def _should_convert_layer(
    name: str,
    module: nn.Module,
    target_layers: Optional[List[str]],
    conv1d_class: Optional[Type]
) -> bool:
    """Determine if a layer should be converted to dendritic."""
    is_linear = isinstance(module, nn.Linear)
    is_conv1d = conv1d_class is not None and isinstance(module, conv1d_class)
    
    if not (is_linear or is_conv1d):
        return False
    
    if target_layers is None:
        return True
        
    return any(pattern in name for pattern in target_layers)


def _create_dendritic_layer(
    module: Union[nn.Linear, Any],  # Union with Any to handle Conv1D
    is_conv1d: bool,
    poly_rank: Union[int, Literal["auto"]],
    init_scale: float,
    device: torch.device,
    dendritic_cls: Type[nn.Module],
    dendritic_kwargs: Dict[str, Any]
) -> Tuple[nn.Module, int]:
    """
    Create and initialize a dendritic layer with proper type handling.
    
    Args:
        module: The original linear or Conv1D module to replace
        is_conv1d: Whether the module is a Conv1D layer
        poly_rank: Either an integer or "auto" for automatic calculation
        init_scale: Initial scale for polynomial pathway
        device: Device to create the dendritic layer on
        dendritic_cls: The dendritic layer class to use
        dendritic_kwargs: Additional keyword arguments for the dendritic layer
        
    Returns:
        Tuple of (dendritic_layer, effective_poly_rank)
    """
    """
    Create and initialize a dendritic layer with proper type handling.
    
    Returns:
        Tuple of (dendritic_layer, effective_poly_rank)
    """
    """Create and initialize a dendritic layer."""
    # Get input/output dimensions with proper type handling
    if is_conv1d:
        # For Conv1D, we need to access weight directly
        weight_tensor = cast(Tensor, module.weight)
        input_dim = weight_tensor.shape[0]  # Conv1D: (out_features, in_features)
        output_dim = weight_tensor.shape[1]
        has_bias = hasattr(module, 'bias') and module.bias is not None
    else:
        # For nn.Linear, use the proper attributes
        linear_module = cast(nn.Linear, module)
        input_dim = linear_module.in_features
        output_dim = linear_module.out_features
        has_bias = linear_module.bias is not None

    # Calculate poly_rank if auto, ensuring we return an int
    if poly_rank == "auto":
        effective_poly_rank = max(4, input_dim // 64)
    else:
        effective_poly_rank = poly_rank

    # Build kwargs for construction
    layer_kwargs = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "poly_rank": effective_poly_rank,
        "init_scale": init_scale,
        "bias": has_bias,
        **dendritic_kwargs
    }

    # Create and initialize dendritic layer
    dendritic = dendritic_cls(**layer_kwargs)
    cast(torch.nn.Module, dendritic).to(device)
    _initialize_dendritic_layer(dendritic, module, is_conv1d, init_scale)
    
    return dendritic, effective_poly_rank


def _initialize_dendritic_layer(
    dendritic: nn.Module,
    original_module: Union[nn.Linear, Any],  # Union with Any to handle Conv1D
    is_conv1d: bool,
    init_scale: float
) -> None:
    """Initialize dendritic layer weights with proper type handling."""
    """Initialize dendritic layer weights based on original module."""
    with torch.no_grad():
        if hasattr(dendritic, "base_linear") and dendritic.__class__.__name__ == "DendriticStack":
            # Handle weight source with proper typing
            weight_tensor = cast(Tensor, original_module.weight)
            src_weight = weight_tensor if not is_conv1d else weight_tensor.t().contiguous()
            
            # Copy original weights to base_linear (Identity Path)
            base_linear = getattr(dendritic, "base_linear")
            if hasattr(base_linear, "weight"):
                base_linear.weight.copy_(src_weight)
            
            # Handle bias if exists
            if hasattr(original_module, "bias") and original_module.bias is not None:
                bias_tensor = cast(Tensor, original_module.bias)
                if hasattr(base_linear, "bias") and base_linear.bias is not None:
                    base_linear.bias.copy_(bias_tensor)
            
            # Zero out new branch (Stack Path)
            if hasattr(dendritic, "layer2") and hasattr(dendritic.layer2, "linear"):
                layer2_linear = getattr(dendritic.layer2, "linear")
                if hasattr(layer2_linear, "weight"):
                    layer2_linear.weight.zero_()
                if hasattr(layer2_linear, "bias") and layer2_linear.bias is not None:
                    layer2_linear.bias.zero_()
                
                # Zero out polynomial components
                for layer_attr in ["layer1", "layer2"]:
                    if hasattr(dendritic, layer_attr):
                        layer = getattr(dendritic, layer_attr)
                        if hasattr(layer, "scale"):
                            layer.scale.fill_(0.0)
                        if hasattr(layer, "diag_scale"):
                            layer.diag_scale.fill_(0.0)
        else:
            # For standard DendriticLayer
            if hasattr(dendritic, "linear") and hasattr(original_module, "weight"):
                linear_layer = getattr(dendritic, "linear")
                weight_tensor = cast(Tensor, original_module.weight)
                src_weight = weight_tensor.t().contiguous() if is_conv1d else weight_tensor
                
                if hasattr(linear_layer, "weight"):
                    linear_layer.weight.copy_(src_weight)
                
                # Handle bias if exists
                if hasattr(original_module, "bias") and original_module.bias is not None:
                    bias_tensor = cast(Tensor, original_module.bias)
                    if hasattr(linear_layer, "bias") and linear_layer.bias is not None:
                        linear_layer.bias.copy_(bias_tensor)
            
            # Initialize scaling parameters with proper type checking
            for param_name in ["scale", "diag_scale"]:
                if hasattr(dendritic, param_name):
                    param = getattr(dendritic, param_name)
                    if isinstance(param, Tensor):
                        param.fill_(init_scale)


def _freeze_linear_pathway(dendritic: nn.Module) -> None:
    """Freeze the linear pathway of a dendritic layer with proper type checking."""
    for component in ["linear", "base_linear"]:
        if hasattr(dendritic, component):
            linear = getattr(dendritic, component)
            if hasattr(linear, "weight"):
                linear.weight.requires_grad = False
            if hasattr(linear, "bias") and linear.bias is not None:
                linear.bias.requires_grad = False


def _get_trainable_parameter_names() -> Set[str]:
    """Return set of parameter names that should be trainable."""
    return {
        "w1", "w2", "w_", "scale", "diag", "poly_", "w_diag",
        "interaction", "attention", "projection"
    }


def _is_parameter_trainable(param_name: str) -> bool:
    """Check if a parameter should be trainable based on its name."""
    trainable_patterns = _get_trainable_parameter_names()
    return any(pattern in param_name for pattern in trainable_patterns)


def _get_module_parent_and_name(model: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    """Get parent module and child name from full module name."""
    if "." in full_name:
        parent_name = full_name.rsplit(".", 1)[0]
        child_name = full_name.split(".")[-1]
        parent = model.get_submodule(parent_name)
    else:
        # Direct child of root module
        parent = model
        child_name = full_name
    return parent, child_name


def _log_conversion_statistics(conversions: List[LayerConversionStats]) -> None:
    """Log detailed statistics about the conversion process."""
    if not conversions:
        return

    total_before = sum(c.linear_params for c in conversions)
    total_after = sum(c.dendritic_params for c in conversions)
    total_trainable = sum(c.trainable_params for c in conversions)

    # Print trainable parameters
    print("\nTrainable parameters after enhancement:")
    for conv in conversions:
        print(f"  {conv.name} | "
              f"Params: {conv.dendritic_params:,} | "
              f"Trainable: {conv.trainable_percentage:.1f}%")

    # Print summary
    print(f"\n{'='*70}")
    print("Dendritic Enhancement Summary")
    print(f"{'='*70}")
    print(f"Converted {len(conversions)} layer(s)")
    print(f"Parameters before:  {total_before:>12,}")
    print(f"Parameters after:   {total_after:>12,}")
    print(f"Added parameters:   {total_after - total_before:>12,} "
          f"(+{(total_after/total_before-1)*100:.2f}%)")
    print(f"Trainable params:   {total_trainable:>12,} "
          f"({total_trainable/total_after*100:.2f}% of enhanced layers)")


def enhance_model_with_dendritic(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    poly_rank: Union[int, Literal["auto"]] = "auto",
    init_scale: float = 1e-6,
    freeze_linear: bool = True,
    verbose: bool = True,
    dendritic_cls: Optional[Type] = None,
    dendritic_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> nn.Module:
    """
    Enhance a pretrained model by replacing linear layers with dendritic versions.

    Args:
        model: PyTorch model to enhance
        target_layers: List of layer name patterns to replace
        poly_rank: 'auto' for input_dim//16, or integer for fixed rank
        init_scale: Initial scale for polynomial pathway
        freeze_linear: Whether to freeze the pretrained linear weights
        verbose: Print conversion statistics
        dendritic_cls: Class to use for replacement
        dendritic_kwargs: Extra keyword arguments for dendritic_cls

    Returns:
        Enhanced model with dendritic layers

    Raises:
        NoLayersConvertedError: If no layers were converted
        ValueError: For invalid input parameters
    """
    # Input validation
    _validate_enhancement_params(model, poly_rank, target_layers)
    
    # Set default dendritic class if not provided
    if dendritic_cls is None:
        from .layers.DendriticLayer import DendriticLayer
        dendritic_cls = DendriticLayer
    
    dendritic_kwargs = dendritic_kwargs or {}
    conv1d_class = _get_conv1d_class()
    device = next(model.parameters()).device
    conversions: List[LayerConversionStats] = []
    replacements: List[Tuple[nn.Module, str, nn.Module]] = []
    dendritic_param_ids: Set[int] = set()

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # First pass: identify and prepare layers for conversion
    for name, module in model.named_modules():
        if not _should_convert_layer(name, module, target_layers, conv1d_class):
            continue

        is_conv1d = conv1d_class is not None and isinstance(module, conv1d_class)
        
        try:
            # Create and initialize dendritic layer
            dendritic, effective_poly_rank = _create_dendritic_layer(
                module=module,
                is_conv1d=is_conv1d,
                poly_rank=poly_rank,
                init_scale=init_scale,
                device=device,
                dendritic_cls=dendritic_cls,
                dendritic_kwargs=dendritic_kwargs
            )

            # Freeze linear pathway if requested
            if freeze_linear:
                _freeze_linear_pathway(dendritic)

            # Mark dendritic parameters as trainable
            for param_name, param in dendritic.named_parameters():
                if _is_parameter_trainable(param_name):
                    param.requires_grad = True
                    dendritic_param_ids.add(id(param))

            # Track conversion statistics
            linear_params = sum(p.numel() for p in module.parameters())
            dendritic_params = sum(p.numel() for p in dendritic.parameters())
            trainable_params = sum(p.numel() for p in dendritic.parameters() if p.requires_grad)

            conversions.append(LayerConversionStats(
                name=name,
                linear_params=linear_params,
                dendritic_params=dendritic_params,
                trainable_params=trainable_params,
                poly_rank=effective_poly_rank
            ))

            # Store replacement info with proper type handling
            parent, child_name = _get_module_parent_and_name(model, name)
            replacements.append((parent, child_name, dendritic))

        except Exception as e:
            logger.error(f"Error converting layer {name}: {str(e)}")
            if verbose:
                print(f"Warning: Failed to convert layer {name}: {str(e)}")
            continue

    if not replacements:
        raise NoLayersConvertedError(
            "No layers were converted. Check target_layers patterns and layer types."
        )

    # Second pass: actually replace modules
    for parent, child_name, new_module in replacements:
        setattr(parent, child_name, new_module)

    # Log conversion statistics
    if verbose:
        _log_conversion_statistics(conversions)

    # Attach configuration to model with proper type handling
    config = {
        "target_layers": target_layers,
        "poly_rank": poly_rank,
        "init_scale": init_scale,
        "freeze_linear": freeze_linear,
        "verbose": verbose,
        "dendritic_cls": dendritic_cls,
        "dendritic_kwargs": dendritic_kwargs,
        **kwargs,
    }
    # Use setattr to avoid mypy issues with dynamic attributes
    setattr(model, "dendritic_config", config)

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


def get_polynomial_stats(model: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Get statistics about polynomial pathways in the model.

    Returns:
        Dictionary with scale values and other statistics for each dendritic layer.
    """
    from .layers.DendriticStack import DendriticStack

    stats: Dict[str, Dict[str, Any]] = {}
    
    for name, module in model.named_modules():
        if not isinstance(module, (DendriticLayer, DendriticStack)):
            continue

        with torch.no_grad():
            # Initialize scale with default value
            scale: float = 0.0
            
            # Get scale from module if it exists
            if hasattr(module, "scale"):
                scale_attr = getattr(module, "scale")
                if isinstance(scale_attr, (int, float)):
                    scale = float(scale_attr)
                elif isinstance(scale_attr, torch.Tensor):
                    scale = scale_attr.item() if scale_attr.numel() == 1 else 0.0

            # Handle DendriticStack layers
            if hasattr(module, "layer1") and hasattr(module.layer1, "scale"):
                layer1_scale = getattr(module.layer1, "scale", 0.0)
                if isinstance(layer1_scale, torch.Tensor) and layer1_scale.numel() == 1:
                    scale = (scale + layer1_scale.item()) / 2
                elif isinstance(layer1_scale, (int, float)):
                    scale = (scale + float(layer1_scale)) / 2

            if hasattr(module, "layer2") and hasattr(module.layer2, "scale"):
                layer2_scale = getattr(module.layer2, "scale", 0.0)
                if isinstance(layer2_scale, torch.Tensor) and layer2_scale.numel() == 1:
                    scale = (scale + layer2_scale.item()) / 2
                elif isinstance(layer2_scale, (int, float)):
                    scale = (scale + float(layer2_scale)) / 2

            # Effective rank calculation
            eff_rank = None
            if hasattr(module, "w1"):
                try:
                    w1 = getattr(module, "w1")
                    if isinstance(w1, torch.Tensor):
                        _, S, _ = torch.svd(w1)
                        eff_rank = (S.sum() ** 2) / (S**2).sum()
                        if isinstance(eff_rank, torch.Tensor):
                            eff_rank = eff_rank.item()
                except Exception as e:
                    logger.warning(f"Failed to calculate effective rank for {name}: {str(e)}")

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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    print(f"Extracted state with dendritic parameters")
    print("Parameter keys:", list(state["state_dict"].keys())[:5], "...")

    # Create new enhanced model with saved state
    new_enhanced = apply_dendritic_state(
        model,
        state
    )

    # Verify same outputs
    test_input = torch.randn(2, 128)
    diff = verify_identity_initialization(enhanced, new_enhanced, test_input)
    print(f"Model output difference after loading state: {diff:.6f}")
