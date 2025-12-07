import torch
import torch.nn as nn
from typing import Optional, List, Union

def enhance_model_with_dendritic(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    poly_rank: Union[int, str] = "auto",
    init_scale: float = 1e-6,
    freeze_linear: bool = True,
    verbose: bool = True,
    dendritic_cls: type = None,
    dendritic_kwargs: dict = None
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
        >>> from DendriticLayer import DendriticLayer, DendriticStack
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
        from layer import DendriticLayer
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
            should_convert = (
                target_layers is None or 
                any(pattern in name for pattern in target_layers)
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
                    bias=has_bias
                )
                layer_kwargs.update(dendritic_kwargs)

                # Create dendritic layer ON THE SAME DEVICE
                dendritic = dendritic_cls(**layer_kwargs).to(device)

                # Custom patching logic for DendriticStack
                with torch.no_grad():
                    if dendritic_cls.__name__ == "DendriticStack":
                        src_weight = module.weight if not is_conv1d else module.weight.t().contiguous()
                        
                        # 1. Copy original weights to base_linear (The Identity Path)
                        if hasattr(dendritic, "base_linear"):
                            dendritic.base_linear.weight.copy_(src_weight)
                            if has_bias and dendritic.base_linear.bias is not None:
                                dendritic.base_linear.bias.copy_(module.bias)
                        
                        # 2. ZERO OUT the new branch (The Stack Path)
                        # If we zero the FINAL layer of the stack, the whole stack outputs 0.
                        # This ensures: Output = Original_Weights(x) + 0
                        if hasattr(dendritic.layer2, "linear"):
                            dendritic.layer2.linear.weight.zero_()
                            if dendritic.layer2.linear.bias is not None:
                                dendritic.layer2.linear.bias.zero_()
                            
                            # Zero out polynomial components of output layer
                            dendritic.layer2.scale.fill_(0.0)
                            if hasattr(dendritic.layer2, "diag_scale"):
                                dendritic.layer2.diag_scale.fill_(0.0)
                        
                        # (Optional) Initialize internal layers to reasonable small values
                        # to aid gradient flow once training starts, even though output is 0
                        # This is handled by default init, so we just ensure the output is killed.

                    else:
                        # Existing logic for DendriticLayer (single layer)
                        if hasattr(dendritic, "linear") and hasattr(module, "weight"):
                            if is_conv1d:
                                dendritic.linear.weight.copy_(module.weight.t().contiguous())
                            else:
                                dendritic.linear.weight.copy_(module.weight)
                            if has_bias:
                                dendritic.linear.bias.copy_(module.bias)
                        if hasattr(dendritic, "scale"):
                            dendritic.scale.fill_(init_scale)

                # Freeze linear pathway logic update
                if freeze_linear:
                    # Case A: DendriticStack (Residual Architecture)
                    # We freeze the 'base_linear' which holds the original weights.
                    # We leave layer1 and layer2 (the adapter) fully trainable.
                    if hasattr(dendritic, "base_linear"):
                        dendritic.base_linear.weight.requires_grad = False
                        if dendritic.base_linear.bias is not None:
                            dendritic.base_linear.bias.requires_grad = False

                    # Case B: DendriticLayer (Direct Replacement)
                    # We freeze the 'linear' part (original weights).
                    # We leave 'w1', 'w2', 'poly_out' etc. trainable.
                    elif hasattr(dendritic, "linear"):
                        dendritic.linear.weight.requires_grad = False
                        if dendritic.linear.bias is not None:
                            dendritic.linear.bias.requires_grad = False
                            
                # Unfreeze dendritic pathway (polynomial) parameters
                for n, p in dendritic.named_parameters():
                    if n.startswith("w1") or n.startswith("w2") or n.startswith("w_out") or n.startswith("scale"):
                        p.requires_grad = True
                        dendritic_param_ids.add(id(p))

                # Track statistics
                linear_params = sum(p.numel() for p in module.parameters())
                dendritic_params = sum(p.numel() for p in dendritic.parameters())
                trainable_params = sum(p.numel() for p in dendritic.parameters() if p.requires_grad)

                conversions.append({
                    'name': name,
                    'linear_params': linear_params,
                    'dendritic_params': dendritic_params,
                    'added_params': dendritic_params - linear_params,
                    'trainable_params': trainable_params,
                    'poly_rank': pr
                })

                # Store replacement info
                if '.' in name:
                    parent_name = name.rsplit('.', 1)[0]
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name)
                else:
                    # Direct child of root module (e.g., 'fc1' in model.fc1)
                    parent = model
                    child_name = name

                replacements.append((parent, child_name, dendritic))

    # Second pass: actually replace modules
    for parent, child_name, new_module in replacements:
        setattr(parent, child_name, new_module)

    # Print trainable parameter names for verification
    print("\nTrainable parameters after enhancement:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n} | shape={tuple(p.shape)}")

    if verbose and conversions:
        total_params_before = sum(c['linear_params'] for c in conversions)
        total_params_after = sum(c['dendritic_params'] for c in conversions)
        total_trainable = sum(c['trainable_params'] for c in conversions)

        print(f"\n{'='*70}")
        print("Dendritic Enhancement Summary")
        print(f"{'='*70}")
        print(f"Converted {len(conversions)} layer(s)")
        print(f"Parameters before:  {total_params_before:>12,}")
        print(f"Parameters after:   {total_params_after:>12,}")
        print(f"Added parameters:   {total_params_after - total_params_before:>12,} "
              f"(+{(total_params_after/total_params_before-1)*100:.2f}%)")
        print(f"\nTrainable params:   {total_trainable:>12,} "
              f"({100*total_trainable/total_params_after:.2f}% of enhanced layers)")

        if freeze_linear:
            print(f"Frozen params:      {total_params_before:>12,} (original linear weights)")

        # Show per-layer breakdown
        print(f"\n{'Layer':<45} {'Rank':<6} {'Added':<12} {'Trainable':<12}")
        print(f"{'-'*70}")
        for conv in conversions[:10]:  # Show first 10
            print(f"{conv['name'][:45]:<45} {conv['poly_rank']:<6} "
                  f"{conv['added_params']:>11,} {conv['trainable_params']:>11,}")

        if len(conversions) > 10:
            print(f"... and {len(conversions) - 10} more layers")

        print(f"{'='*70}\n")

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
    
    Returns dict with scale values and effective ranks for each dendritic layer.
    """
    from layer import DendriticLayer
    
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, DendriticLayer):
            with torch.no_grad():
                scale = module.scale.item()
                
                # Effective rank via singular values
                try:
                    U, S, V = torch.svd(module.w1)
                    eff_rank = (S.sum() ** 2) / (S ** 2).sum()
                    eff_rank = eff_rank.item()
                except:
                    eff_rank = None
                
                stats[name] = {
                    'scale': scale,
                    'eff_rank': eff_rank,
                    'poly_rank': module.poly_rank
                }
    
    return stats


# Example usage
if __name__ == "__main__":
    print("Example: Enhancing a simple model\n")
    
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
    
    model = SimpleModel()
    
    # Enhance only fc1 and fc2, leave fc3 as standard linear
    model_enhanced = enhance_model_with_dendritic(
        model,
        target_layers=['fc1', 'fc2'],
        poly_rank=8,
        freeze_linear=True
    )

    # Verify identity initialization
    test_input = torch.randn(4, 128)
    diff = verify_identity_initialization(model, model_enhanced, test_input)
    print(f"Max output difference (DendriticLayer): {diff:.2e} (should be ~1e-6 or less)\n")

    # Check what's trainable
    trainable = sum(p.numel() for p in model_enhanced.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_enhanced.parameters())
    print(f"Trainable parameters (DendriticLayer): {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Now test with DendriticStack
    from layer import DendriticStack
    model_stack = SimpleModel()
    model_enhanced_stack = enhance_model_with_dendritic(
        model_stack,
        target_layers=['fc1', 'fc2'],
        poly_rank=8,
        freeze_linear=True,
        dendritic_cls=DendriticStack,
        dendritic_kwargs={"bottleneck_dim": 64, "dropout": 0.1}
    )
    diff_stack = verify_identity_initialization(model_stack, model_enhanced_stack, test_input)
    print(f"Max output difference (DendriticStack): {diff_stack:.2e} (should be ~1e-6 or less)\n")
    trainable_stack = sum(p.numel() for p in model_enhanced_stack.parameters() if p.requires_grad)
    total_stack = sum(p.numel() for p in model_enhanced_stack.parameters())
    print(f"Trainable parameters (DendriticStack): {trainable_stack:,} / {total_stack:,} ({100*trainable_stack/total_stack:.2f}%)")