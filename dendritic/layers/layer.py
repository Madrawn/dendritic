"""
Dendritic computation layers for neural networks.

These layers add learnable polynomial (quadratic) features to standard linear
transformations, inspired by dendritic nonlinear integration in biological neurons.

This module serves as a backward-compatible wrapper that imports functionality
from the refactored modules:
- `sanity.py`: Quick sanity checks and performance tests
- `benchmark.py`: Comprehensive diagnostic benchmarks

Benchmark findings:
- Asymmetric formulation (W₁x ⊙ W₂x) optimizes much better than symmetric (Px)²
- poly_rank ≈ target_rank achieves perfect fit with proper sample count
- Need ~2x parameters worth of samples for generalization
- Explicit diagonal pathway helps when x_i² terms dominate

Example usage:
    # Drop-in replacement for nn.Linear
    layer = DendriticLayer(256, 128, poly_rank=16)

    # In a transformer MLP
    mlp = nn.Sequential(
        DendriticLayer(embed_dim, hidden_dim, poly_rank=embed_dim // 16),
        nn.GELU(),
        nn.Linear(hidden_dim, embed_dim),
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

from dendritic.layers.DendriticLayer import DendriticLayer
from dendritic.layers.DendriticMLP import DendriticMLP
from dendritic.layers.DendriticStack import DendriticStack

# Import all functions from the refactored modules for backward compatibility
from dendritic.layers.sanity import (
    _test,
    check_true_capacity,
    performance_test,
    test_shapes,
    test_gradient_flow,
    test_parameter_count,
)

from dendritic.layers.benchmark import (
    create_rank_k_target,
    run_training_benchmark,
    diagnostic_benchmark,
    evaluate_rank_scaling,
    test_structure_responsiveness,
    assess_cubic_feature_contributions,
    evaluate_model_generalization,
)

# Provide backward compatibility alias for make_rank_k_target
make_rank_k_target = create_rank_k_target

# Re-export all public functions
__all__ = [
    # Layer classes
    "DendriticLayer",
    "DendriticMLP",
    "DendriticStack",
    # Sanity check functions
    "_test",
    "check_true_capacity",
    "performance_test",
    "test_shapes",
    "test_gradient_flow",
    "test_parameter_count",
    # Benchmark functions
    "create_rank_k_target",
    "make_rank_k_target",  # Backward compatibility alias
    "run_training_benchmark",
    "diagnostic_benchmark",
    "evaluate_rank_scaling",
    "test_structure_responsiveness",
    "assess_cubic_feature_contributions",
    "evaluate_model_generalization",
]


if __name__ == "__main__":
    # Maintain the original main block for backward compatibility
    _test()
    check_true_capacity()
    performance_test()
    diagnostic_benchmark(n_batches=2500, batch_size=100)
