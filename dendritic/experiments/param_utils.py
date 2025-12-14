# dendritic/experiments/param_utils.py
"""Utilities for parameter counting and matching across architectures."""

from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import math

import torch
import torch.nn as nn


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
    # Linear pathway
    linear_params = input_dim * output_dim + (output_dim if bias else 0)
    
    # Cross-term pathway
    cross_params = 2 * poly_rank * input_dim  # W1, W2
    cross_params += output_dim * poly_rank     # poly_out
    cross_params += 1                          # scale
    
    # Diagonal pathway
    diag_params = 0
    if diag_rank > 0:
        diag_params = diag_rank * input_dim    # W_diag_in
        diag_params += output_dim * diag_rank  # W_diag_out
        diag_params += 1                       # diag_scale
    
    return linear_params + cross_params + diag_params


def count_dendritic_stack_params(
    input_dim: int,
    output_dim: int,
    poly_rank: int,
    bottleneck_dim: Optional[int] = None,
    diag_rank: Optional[int] = None,
    bias: bool = True,
    preserve_linear_path: bool = True
) -> int:
    """
    Calculate exact parameter count for a DendriticStack.
    """
    if bottleneck_dim is None:
        bottleneck_dim = poly_rank * 2
    
    if diag_rank is None:
        diag_rank = max(4, poly_rank // 4)
    
    # Layer 1: input_dim -> bottleneck_dim
    layer1_params = count_dendritic_layer_params(
        input_dim, bottleneck_dim, poly_rank, diag_rank, bias=True
    )
    
    # Layer 2: bottleneck_dim -> output_dim  
    layer2_params = count_dendritic_layer_params(
        bottleneck_dim, output_dim, poly_rank, diag_rank, bias=bias
    )
    
    # Base linear path (if preserved)
    base_linear_params = 0
    if preserve_linear_path:
        base_linear_params = input_dim * output_dim + (output_dim if bias else 0)
    
    return layer1_params + layer2_params + base_linear_params


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