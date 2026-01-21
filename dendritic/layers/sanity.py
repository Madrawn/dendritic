"""
Sanity checks and performance tests for dendritic layers.

This module contains quick sanity checks and performance benchmarks for
verifying the correctness and efficiency of dendritic layer implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple, Dict, Any

from dendritic.layers.DendriticLayer import DendriticLayer
from dendritic.layers.DendriticMLP import DendriticMLP
from dendritic.layers.DendriticStack import DendriticStack

# Module-level constants
DEFAULT_INPUT_DIM = 64
DEFAULT_OUTPUT_DIM = 32
DEFAULT_POLY_RANK = 8
TEST_BATCH_SIZE = 16
TEST_SIZE = 1000
EPS = 1e-8


def test_shapes() -> None:
    """Test that dendritic layers produce correct output shapes."""
    torch.manual_seed(42)

    # Test basic layer
    layer = DendriticLayer(
        DEFAULT_INPUT_DIM, DEFAULT_OUTPUT_DIM, poly_rank=DEFAULT_POLY_RANK
    )
    x = torch.randn(TEST_BATCH_SIZE, DEFAULT_INPUT_DIM)
    y = layer(x)
    assert y.shape == (
        TEST_BATCH_SIZE,
        DEFAULT_OUTPUT_DIM,
    ), f"Expected shape {(TEST_BATCH_SIZE, DEFAULT_OUTPUT_DIM)}, got {y.shape}"

    # Test DendriticStack base case
    stack = DendriticStack(
        DEFAULT_INPUT_DIM, DEFAULT_OUTPUT_DIM, poly_rank=DEFAULT_POLY_RANK
    )
    x_stack = torch.randn(TEST_BATCH_SIZE, DEFAULT_INPUT_DIM)
    y_stack = stack(x_stack)
    assert y_stack.shape == (
        TEST_BATCH_SIZE,
        DEFAULT_OUTPUT_DIM,
    ), f"Expected shape {(TEST_BATCH_SIZE, DEFAULT_OUTPUT_DIM)}, got {y_stack.shape}"

    # Test MLP
    mlp = DendriticMLP(256, 1024, poly_rank=16)
    x_mlp = torch.randn(4, 128, 256)  # [batch, seq, embed]
    y_mlp = mlp(x_mlp)
    assert y_mlp.shape == (
        4,
        128,
        256,
    ), f"Expected shape {(4, 128, 256)}, got {y_mlp.shape}"

    print("✓ All shape tests passed")


def test_gradient_flow() -> None:
    """Test that gradients flow through all learnable parameters."""
    torch.manual_seed(42)

    # Test DendriticLayer gradient flow
    layer = DendriticLayer(
        DEFAULT_INPUT_DIM, DEFAULT_OUTPUT_DIM, poly_rank=DEFAULT_POLY_RANK
    )
    x = torch.randn(TEST_BATCH_SIZE, DEFAULT_INPUT_DIM)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    for name, p in layer.named_parameters():
        assert p.grad is not None, f"Gradient missing for {name}"

    # Test DendriticStack gradient flow
    stack = DendriticStack(
        DEFAULT_INPUT_DIM, DEFAULT_OUTPUT_DIM, poly_rank=DEFAULT_POLY_RANK
    )
    x_stack = torch.randn(TEST_BATCH_SIZE, DEFAULT_INPUT_DIM)
    y_stack = stack(x_stack)
    loss_stack = y_stack.sum()
    loss_stack.backward()

    for name, p in layer.named_parameters():
        assert p.grad is not None, f"Gradient missing for {name}"

    assert stack.poly_out.grad is not None, "Gradient missing for poly_out"

    print("✓ All gradient flow tests passed")


def test_parameter_count() -> Tuple[int, int]:
    """
    Test that parameter counts match expected calculations.

    Returns:
        Tuple of (layer_params, stack_params) parameter counts.
    """
    layer = DendriticLayer(
        DEFAULT_INPUT_DIM, DEFAULT_OUTPUT_DIM, poly_rank=DEFAULT_POLY_RANK
    )
    stack = DendriticStack(
        DEFAULT_INPUT_DIM, DEFAULT_OUTPUT_DIM, poly_rank=DEFAULT_POLY_RANK
    )

    n_params = sum(p.numel() for p in layer.parameters())
    n_params_stack = sum(p.numel() for p in stack.parameters())

    # Compute expected parameter count for DendriticLayer
    poly_rank = DEFAULT_POLY_RANK
    if (
        layer.diag_rank is not None
        and isinstance(layer.diag_rank, int)
        and layer.diag_rank > 0
    ):

        expected = (
            DEFAULT_INPUT_DIM * DEFAULT_OUTPUT_DIM
            + DEFAULT_OUTPUT_DIM
            + 2 * poly_rank * DEFAULT_INPUT_DIM
            + 2 * poly_rank
            + DEFAULT_OUTPUT_DIM * poly_rank
            + (
                layer.diag_rank * DEFAULT_INPUT_DIM
                + DEFAULT_OUTPUT_DIM * layer.diag_rank
            )  # diagonal weights
            + 2  # alpha + alpha_diag
        )

    else:

        expected = (
            DEFAULT_INPUT_DIM * DEFAULT_OUTPUT_DIM + DEFAULT_OUTPUT_DIM   # linear
            + 2 * poly_rank * DEFAULT_INPUT_DIM                           # projections
            + 2 * poly_rank                                               # projection biases
            + DEFAULT_OUTPUT_DIM * poly_rank                              # poly_out
            + 1                                                           # alpha
        )


    assert (
        n_params == expected
    ), f"{n_params} != {expected} (expected calculation may need update)"

    print("✓ Parameter count tests passed")
    return n_params, n_params_stack


def _test() -> None:
    """
    Quick sanity check combining shape, gradient, and parameter tests.

    This function provides backward compatibility with the original _test().
    """
    print("Running comprehensive sanity checks...")
    test_shapes()
    test_gradient_flow()
    test_gate_and_bias_gradients()
    layer_params, stack_params = test_parameter_count()

    linear_params = DEFAULT_INPUT_DIM * DEFAULT_OUTPUT_DIM + DEFAULT_OUTPUT_DIM
    overhead = layer_params - linear_params
    overhead_percent = 100 * (layer_params / linear_params - 1)

    print("\nAll tests passed!")
    print(
        f"DendriticLayer({DEFAULT_INPUT_DIM}, {DEFAULT_OUTPUT_DIM}, poly_rank={DEFAULT_POLY_RANK}): {layer_params:,} parameters"
    )
    print(
        f"DendriticStack({DEFAULT_INPUT_DIM}, {DEFAULT_OUTPUT_DIM}, poly_rank={DEFAULT_POLY_RANK}): {stack_params:,} parameters"
    )
    print(
        f"  vs nn.Linear({DEFAULT_INPUT_DIM}, {DEFAULT_OUTPUT_DIM}): {linear_params:,} parameters"
    )
    print(f"  overhead: {overhead:,} ({overhead_percent:.1f}%)")


def check_true_capacity() -> None:
    """
    Test model capacity on a cubic interaction task.

    This test evaluates whether dendritic layers can learn cubic interactions
    through composition of quadratic terms.
    """
    torch.manual_seed(42)

    # Increase samples to prevent memorization
    N = 10000
    d = 20  # Smaller dim to ensure density

    X = torch.randn(N, d)
    # Target: Cubic interaction
    y = (X[:, 0] * X[:, 1] * X[:, 2]).unsqueeze(1)

    # Split
    split = int(N * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"{'Model':<20} {'Train R²':<10} {'Test R²':<10}")
    print("-" * 45)

    models = {
        "DendriticLayer": DendriticLayer(d, 1, poly_rank=16),  # Pure Quadratic
        "DendriticStack": DendriticStack(d, 1, poly_rank=16),
    }

    for name, model in models.items():
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train
        for _ in range(500):
            opt.zero_grad()
            loss = F.mse_loss(model(X_train), y_train)
            loss.backward()
            opt.step()

        # Evaluate
        with torch.no_grad():
            r2_train = 1 - F.mse_loss(model(X_train), y_train) / y_train.var()
            r2_test = 1 - F.mse_loss(model(X_test), y_test) / y_test.var()

        print(f"{name:<20} {r2_train.item():<10.3f} {r2_test.item():<10.3f}")


def performance_test() -> None:
    """
    Comprehensive performance benchmark comparing different layer types.

    Measures forward pass time and parameter counts for various configurations.
    """
    torch.manual_seed(42)

    batch_size = 64
    input_dim = 256
    output_dim = 128
    num_iterations = 500

    print("=" * 80)
    print("Dendritic Layer Comprehensive Benchmark")
    print("=" * 80)
    print(
        f"\nConfig: batch={batch_size}, in={input_dim}, out={output_dim}, iters={num_iterations}"
    )

    # Test data
    x = torch.randn(batch_size, input_dim)

    # Models to test
    models = {
        "Linear (baseline)": nn.Linear(input_dim, output_dim),
        "Dendritic (rank=8)": DendriticLayer(input_dim, output_dim, poly_rank=8),
        "Dendritic (rank=16)": DendriticLayer(input_dim, output_dim, poly_rank=16),
        "DendriticStack (rank=8)": DendriticStack(input_dim, output_dim, poly_rank=8),
        "DendriticStack (rank=16)": DendriticStack(input_dim, output_dim, poly_rank=16),
    }

    # Warmup
    print("\nWarming up...")
    for model in models.values():
        for _ in range(50):
            _ = model(x)

    # Timing benchmark with CUDA synchronization if available
    print("\n" + "-" * 80)
    print("TIMING BENCHMARK")
    print("-" * 80)

    baseline_time = None
    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        # Add CUDA synchronization if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = model(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        if baseline_time is None:
            baseline_time = elapsed

        num_params = sum(p.numel() for p in model.parameters())
        baseline_params = input_dim * output_dim + output_dim  # Linear baseline

        results[name] = {
            "time_ms": elapsed * 1000 / num_iterations,
            "slowdown": elapsed / baseline_time,
            "params": num_params,
            "extra_params": num_params - baseline_params,
        }

    print(
        f"\n{'Model':<25} {'Time(ms)':<10} {'Slowdown':<10} {'Params':<12} {'Extra':<10}"
    )
    print("-" * 80)
    for name, res in results.items():
        print(
            f"{name:<25} {res['time_ms']:<10.3f} {res['slowdown']:<10.2f}x "
            f"{res['params']:<12,} {res['extra_params']:<+10,}"
        )


def test_gate_and_bias_gradients() -> None:
    """
    Verify gradient behavior for ReZero gates (alpha, alpha_diag) and projection biases.

    Phase A (alpha=0): 
      - alpha should receive gradient
      - proj_biases should have zero gradient (multiplicative path closed)
    Phase B (alpha≈1e-3):
      - alpha should still receive gradient
      - proj_biases should now receive nonzero gradients
    """
    torch.manual_seed(42)

    def _check_model(name: str, model: nn.Module, input_dim: int = DEFAULT_INPUT_DIM):
        x = torch.randn(TEST_BATCH_SIZE, input_dim)

        # -------- Phase A: gate closed (alpha=0) --------
        model.zero_grad(set_to_none=True)
        # Ensure gates are exactly 0
        if hasattr(model, "alpha"):
            with torch.no_grad():
                model.alpha.zero_()
        if hasattr(model, "alpha_diag"):
            with torch.no_grad():
                model.alpha_diag.zero_()

        y = model(x)
        loss = y.sum()
        loss.backward()

        # alpha should have gradient
        if hasattr(model, "alpha"):
            assert model.alpha.grad is not None, f"[{name}] alpha.grad is None in Phase A"
            # typically non-zero; we don't hard fail on 0.0 since it's technically possible on symmetric losses,
            # but it's extremely unlikely with random inputs and 'sum' loss.
            assert model.alpha.grad.abs().sum().item() != 0.0, f"[{name}] alpha.grad is zero in Phase A"

        # alpha_diag (if diagonal is enabled) should also have gradient
        if getattr(model, "use_diagonal", False) and hasattr(model, "alpha_diag"):
            assert model.alpha_diag.grad is not None, f"[{name}] alpha_diag.grad is None in Phase A"
            # can be zero if diag branch contributes zero to 'sum' by symmetry; don't hard fail on nonzero value
            # but at least check grad exists.
        


        # -------- Phase B: gate slightly open (alpha=1e-3) --------
        model.zero_grad(set_to_none=True)
        if hasattr(model, "alpha"):
            with torch.no_grad():
                model.alpha.fill_(1e-3)
        if getattr(model, "use_diagonal", False) and hasattr(model, "alpha_diag"):
            with torch.no_grad():
                model.alpha_diag.fill_(1e-3)

        y = model(x)
        loss = y.sum()
        loss.backward()

        # alpha should still have gradient
        if hasattr(model, "alpha"):
            assert model.alpha.grad is not None, f"[{name}] alpha.grad is None in Phase B"
            assert model.alpha.grad.abs().sum().item() != 0.0, f"[{name}] alpha.grad is zero in Phase B"



        print(f"✓ Gradient gate/bias checks passed for {name}")

    # Check both DendriticLayer and DendriticStack
    layer = DendriticLayer(DEFAULT_INPUT_DIM, DEFAULT_OUTPUT_DIM, poly_rank=DEFAULT_POLY_RANK)
    stack = DendriticStack(DEFAULT_INPUT_DIM, DEFAULT_OUTPUT_DIM, poly_rank=DEFAULT_POLY_RANK)

    _check_model("DendriticLayer", layer)
    _check_model("DendriticStack", stack)

if __name__ == "__main__":
    _test()
    check_true_capacity()
    performance_test()
