# dendritic/experiments/param_utils.py
"""Utilities for parameter counting and matching across architectures."""

from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

import torch.nn as nn

from .PretrainingConfig import PretrainingConfig


@dataclass
class ParamBreakdown:
    """Detailed parameter breakdown for a model or layer."""
    total: int
    trainable: int
    frozen: int
    by_component: Dict[str, int]
    
    def __repr__(self) -> str:
        lines = [
            f"Total:     {self.total:,}",
            f"Trainable: {self.trainable:,} ({100*self.trainable/self.total:.2f}%)",
            f"Frozen:    {self.frozen:,}",
            "\nBy component:"
        ]
        for name, count in sorted(self.by_component.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count:,}")
        return "\n".join(lines)


def count_parameters(model: nn.Module, prefix: str = "") -> ParamBreakdown:
    """
    Count parameters with detailed breakdown.
    
    Args:
        model: PyTorch model
        prefix: Optional prefix for component names
        
    Returns:
        ParamBreakdown with total, trainable, frozen, and per-component counts
    """
    total = 0
    trainable = 0
    by_component: Dict[str, int] = {}
    
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
        
        # Group by top-level component
        component = name.split(".")[0] if "." in name else name
        full_name = f"{prefix}.{component}" if prefix else component
        by_component[full_name] = by_component.get(full_name, 0) + count
    
    return ParamBreakdown(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
        by_component=by_component
    )


def count_dendritic_layer_params(
    input_dim: int,
    output_dim: int,
    poly_rank: int,
    diag_rank: int,
    bias: bool = True
) -> int:
    """
    Calculate exact parameter count for a DendriticLayer.
    
    Parameters:
    - Linear: input_dim * output_dim + (output_dim if bias)
    - W1, W2: 2 * poly_rank * input_dim
    - poly_out: output_dim * poly_rank
    - scale: 1
    - W_diag_in: diag_rank * input_dim (if diag_rank > 0)
    - W_diag_out: output_dim * diag_rank (if diag_rank > 0)
    - diag_scale: 1 (if diag_rank > 0)
    """
    from dendritic.layers.DendriticLayer import DendriticLayer
    return DendriticLayer.parameter_count(
        input_dim=input_dim,
        output_dim=output_dim,
        poly_rank=poly_rank,
        independent_inputs=False,
        diag_rank=diag_rank,
        bias=bias,
    )


def count_dendritic_stack_params(
    input_dim: int,
    output_dim: int,
    poly_rank: int,
    diag_rank: Optional[int] = None,
    bias: bool = True,
    preserve_linear_path: bool = True,
    poly_degree: int = 3,
    independent_inputs: bool = False
) -> int:
    """
    Calculate exact parameter count for a DendriticStack (new architecture).
    
    The new DendriticStack consists of:
    - Linear pathway (preserved for initialization)
    - Degree-k polynomial pathway with k = poly_degree projections
    - Optional diagonal pathway
    
    Parameters:
        input_dim, output_dim: layer dimensions
        poly_rank: rank of polynomial projections
        diag_rank: diagonal rank; if None or "auto", computed based on independent_inputs
        bias: whether linear pathway includes bias
        preserve_linear_path: whether to include linear pathway parameters
        poly_degree: number of projection matrices (k)
        independent_inputs: if True, diag_rank = poly_rank; else diag_rank = max(4, poly_rank // 4)
    """
    from dendritic.layers.DendriticStack import DendriticStack
    return DendriticStack.parameter_count(
        input_dim=input_dim,
        output_dim=output_dim,
        poly_rank=poly_rank,
        poly_degree=poly_degree,
        independent_inputs=independent_inputs,
        diag_rank=diag_rank if diag_rank is not None else "auto",
        bias=bias,
        include_linear=preserve_linear_path,
    )


def count_lora_params(
    target_modules: List[Tuple[int, int]],  # List of (in_features, out_features)
    rank: int,
    use_bias: bool = False
) -> int:
    """
    Calculate LoRA parameter count.
    
    LoRA adds two low-rank matrices per adapted layer:
    - A: (in_features, rank)
    - B: (rank, out_features)
    
    Total per layer: rank * (in_features + out_features)
    """
    total = 0
    for in_features, out_features in target_modules:
        total += rank * (in_features + out_features)
        if use_bias:
            total += out_features
    return total


def calculate_matching_lora_rank(
    dendritic_trainable_params: int,
    target_modules: List[Tuple[int, int]]
) -> int:
    """
    Calculate LoRA rank that matches dendritic trainable parameter count.
    
    Solves: rank * sum(in_i + out_i) = dendritic_trainable_params
    """
    total_dim_sum = sum(in_f + out_f for in_f, out_f in target_modules)
    rank = dendritic_trainable_params / total_dim_sum
    return max(1, round(rank))


def calculate_matching_mlp_hidden_dim(
    target_params: int,
    embed_dim: int,
    num_layers: int,
    other_params: int  # Params from attention, embeddings, etc.
) -> int:
    """
    Calculate MLP hidden dimension to match target parameter count.
    
    Standard MLP params per layer:
    - fc1: embed_dim * hidden_dim + hidden_dim (bias)
    - fc2: hidden_dim * embed_dim + embed_dim (bias)
    Total per layer: 2 * embed_dim * hidden_dim + hidden_dim + embed_dim
    
    Solving for hidden_dim:
    mlp_params = target_params - other_params
    mlp_params = num_layers * (2 * embed_dim * hidden_dim + hidden_dim + embed_dim)
    """
    mlp_budget = target_params - other_params
    per_layer = mlp_budget / num_layers
    
    # 2 * embed_dim * h + h + embed_dim = per_layer
    # h * (2 * embed_dim + 1) = per_layer - embed_dim
    hidden_dim = (per_layer - embed_dim) / (2 * embed_dim + 1)
    
    return max(1, round(hidden_dim))


def verify_param_match(
    model_a: nn.Module,
    model_b: nn.Module,
    tolerance: float = 0.02,
    trainable_only: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify two models have matching parameter counts within tolerance.
    
    Args:
        model_a, model_b: Models to compare
        tolerance: Acceptable relative difference (default 2%)
        trainable_only: Only compare trainable parameters
        
    Returns:
        (is_matched, details_dict)
    """
    breakdown_a = count_parameters(model_a)
    breakdown_b = count_parameters(model_b)
    
    if trainable_only:
        count_a = breakdown_a.trainable
        count_b = breakdown_b.trainable
    else:
        count_a = breakdown_a.total
        count_b = breakdown_b.total
    
    diff = abs(count_a - count_b)
    rel_diff = diff / max(count_a, count_b)
    
    details = {
        "model_a_params": count_a,
        "model_b_params": count_b,
        "absolute_diff": diff,
        "relative_diff": rel_diff,
        "breakdown_a": breakdown_a,
        "breakdown_b": breakdown_b
    }
    
    return rel_diff <= tolerance, details


def calculate_mlp_params_dendritic_stack(
    embed_dim: int,
    hidden_dim: int,
    poly_rank: int,
    diag_rank: Optional[int] = None
) -> int:
    """Calculate DendriticStack MLP parameters."""
    # DendriticStack parameters (new architecture)
    stack_params = count_dendritic_stack_params(
        embed_dim, hidden_dim, poly_rank, diag_rank=diag_rank, bias=True, preserve_linear_path=True
    )
    # Standard fc2
    fc2_params = hidden_dim * embed_dim + embed_dim
    return stack_params + fc2_params


def calculate_mlp_params_dendritic(
    embed_dim: int,
    hidden_dim: int,
    poly_rank: int,
    diag_rank: Optional[int] = None
) -> int:
    """Calculate DendriticMLP parameters."""
    if diag_rank is None:
        diag_rank = max(4, poly_rank // 4)

    # DendriticLayer (fc1)
    from .param_utils import count_dendritic_layer_params
    dendritic_fc1 = count_dendritic_layer_params(
        embed_dim, hidden_dim, poly_rank, diag_rank, bias=True
    )

    # Standard fc2
    fc2_params = hidden_dim * embed_dim + embed_dim

    return dendritic_fc1 + fc2_params


def calculate_mlp_params_baseline(embed_dim: int, hidden_dim: int) -> int:
    """Calculate standard MLP parameters."""
    # fc1: embed_dim * hidden_dim + hidden_dim
    # fc2: hidden_dim * embed_dim + embed_dim
    return 2 * embed_dim * hidden_dim + hidden_dim + embed_dim


def calculate_non_mlp_params(config: PretrainingConfig) -> int:
    """Calculate parameters outside MLP (embeddings, attention, layer norms)."""
    embed_dim = config.embed_dim
    vocab_size = config.vocab_size
    num_layers = config.num_layers
    num_heads = config.num_heads
    max_seq_len = config.max_seq_len

    # Token embeddings (shared with output head)
    tok_emb_params = vocab_size * embed_dim

    # Position embeddings
    pos_emb_params = max_seq_len * embed_dim

    # Per-layer non-MLP params
    per_layer_params = 0

    # LayerNorm x2: 2 * 2 * embed_dim (weight + bias)
    per_layer_params += 4 * embed_dim

    # Attention: Q, K, V projections + output projection
    # Q, K, V: 3 * (embed_dim * embed_dim + embed_dim)
    # Out: embed_dim * embed_dim + embed_dim
    attn_params = 4 * (embed_dim * embed_dim + embed_dim)
    per_layer_params += attn_params

    # Final layer norm
    final_ln_params = 2 * embed_dim

    total = (
        tok_emb_params +
        pos_emb_params +
        num_layers * per_layer_params +
        final_ln_params
    )

    return total


def find_matching_hidden_dims(config: PretrainingConfig) -> Tuple[int, int, int]:
    """
    Find hidden dimensions that give equal total parameters for all three variants.

    Returns:
        (baseline_hidden_dim, dendritic_hidden_dim, stack_hidden_dim)
    """
    embed_dim = config.embed_dim
    num_layers = config.num_layers
    poly_rank = config.poly_rank

    non_mlp_params = calculate_non_mlp_params(config)

    # Standard ratio is 4x embed_dim
    baseline_hidden = 4 * embed_dim
    baseline_mlp_per_layer = calculate_mlp_params_baseline(embed_dim, baseline_hidden)
    baseline_total = non_mlp_params + num_layers * baseline_mlp_per_layer

    # Target total params
    target_total = baseline_total
    target_mlp_budget = target_total - non_mlp_params
    target_per_layer = target_mlp_budget / num_layers

    # Binary search for dendritic hidden_dim
    diag_rank = max(4, poly_rank // 4)

    def dendritic_mlp_params(h: int) -> int:
        return calculate_mlp_params_dendritic(embed_dim, h, poly_rank, diag_rank)

    def stack_mlp_params(h: int) -> int:
        return calculate_mlp_params_dendritic_stack(embed_dim, h, poly_rank, diag_rank)

    # Search for dendritic hidden_dim
    lo, hi = embed_dim, 8 * embed_dim
    while lo < hi:
        mid = (lo + hi) // 2
        params = dendritic_mlp_params(mid)
        if params < target_per_layer:
            lo = mid + 1
        else:
            hi = mid
    dendritic_hidden = lo

    # Fine-tune dendritic hidden_dim
    best_dendritic = dendritic_hidden
    best_diff = abs(dendritic_mlp_params(dendritic_hidden) - target_per_layer)
    for h in [dendritic_hidden - 1, dendritic_hidden, dendritic_hidden + 1]:
        if h > 0:
            diff = abs(dendritic_mlp_params(h) - target_per_layer)
            if diff < best_diff:
                best_diff = diff
                best_dendritic = h

    # Search for stack hidden_dim
    lo, hi = embed_dim, 8 * embed_dim
    while lo < hi:
        mid = (lo + hi) // 2
        params = stack_mlp_params(mid)
        if params < target_per_layer:
            lo = mid + 1
        else:
            hi = mid
    stack_hidden = lo

    # Fine-tune stack hidden_dim
    best_stack = stack_hidden
    best_diff = abs(stack_mlp_params(stack_hidden) - target_per_layer)
    for h in [stack_hidden - 1, stack_hidden, stack_hidden + 1]:
        if h > 0:
            diff = abs(stack_mlp_params(h) - target_per_layer)
            if diff < best_diff:
                best_diff = diff
                best_stack = h

    return baseline_hidden, best_dendritic, best_stack