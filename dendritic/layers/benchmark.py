"""
Diagnostic benchmarks for dendritic layers.

This module contains comprehensive benchmarking functions for evaluating
the capabilities, limitations, and performance characteristics of dendritic layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from typing import Callable, Dict, List, Tuple, Optional, Any

from dendritic.layers.DendriticLayer import DendriticLayer
from dendritic.layers.DendriticStack import DendriticStack

# Module-level constants
DEFAULT_INPUT_DIM = 64
DEFAULT_OUTPUT_DIM = 1
TEST_SIZE = 1000
EPS = 1e-8


# --- helper to freeze pairs once ---
def make_rank_k_target(k: int, d: int, seed: int = 42, normalize: bool = True):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(d, generator=g)[: 2 * k]
    pairs = perm.view(k, 2)  # (k,2)
    norm = (k**0.5) if normalize else 1.0

    def target_fn(X: torch.Tensor) -> torch.Tensor:
        i, j = pairs[:, 0], pairs[:, 1]
        products = X[:, i] * X[:, j]  # (batch, k)
        return products.sum(dim=1, keepdim=True) / norm

    return target_fn, pairs


def create_rank_k_target(X: torch.Tensor, k: int, seed: int = 42) -> torch.Tensor:
    """
    Create a target that sums k independent pairwise products of input features.

    Args:
        X: Input tensor of shape (batch, d).
        k: Number of disjoint feature pairs to use.
        seed: Random seed for selecting the pairs.

    Returns:
        Tensor of shape (batch, 1) containing the normalized sum of products.

    Raises:
        ValueError: If 2*k > d (not enough features to form k disjoint pairs).
    """
    d = X.shape[1]
    if 2 * k > d:
        raise ValueError(f"Cannot select {k} disjoint pairs from {d} features")

    torch.manual_seed(seed)
    perm = torch.randperm(d)[: 2 * k]
    pairs = perm.reshape(k, 2)  # (k, 2)
    i, j = pairs[:, 0], pairs[:, 1]  # each shape (k,)

    # Vectorized computation
    products = X[:, i] * X[:, j]  # (batch, k)
    result = products.sum(dim=1, keepdim=True)  # (batch, 1)

    # Normalize with a small epsilon to avoid division by zero
    result_std = result.std()
    if result_std == 0:
        result_std = EPS
    result = result / (result_std + EPS)

    return result


def run_training_benchmark(
    model_fn: Callable[[], nn.Module],
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    n_batches: int,
    batch_size: int,
    lr: float = 0.01,
    input_dim: int = DEFAULT_INPUT_DIM,
) -> float:
    """
    Train a model on a synthetic target and return the R² score on a fixed test set.

    Args:
        model_fn: Function that returns a fresh model instance.
        target_fn: Function that takes input tensor and returns target tensor.
        n_batches: Number of training batches.
        batch_size: Size of each training batch.
        lr: Learning rate for Adam optimizer.
        input_dim: Input dimension.

    Returns:
        R² score on a fixed test set.
    """
    model = model_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_batches):
        # X_batch = torch.randn(batch_size, input_dim)
        # # Normalize with epsilon to avoid division by zero
        # X_batch_std = X_batch.std()
        # if X_batch_std == 0:
        # X_batch_std = EPS
        # X_batch = X_batch / (X_batch_std + EPS)

        X_batch = torch.randn(batch_size, input_dim)  # already i.i.d. N(0,1)

        y_batch = target_fn(X_batch)
        optimizer.zero_grad()
        loss = F.mse_loss(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    # Evaluation on fixed test set
    X_test = torch.randn(TEST_SIZE, input_dim)
    X_test_std = X_test.std()
    if X_test_std == 0:
        X_test_std = EPS
    X_test = X_test / (X_test_std + EPS)

    y_test = target_fn(X_test)

    with torch.inference_mode():
        pred_test = model(X_test)
        ss_res = ((pred_test - y_test) ** 2).sum().item()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return r2


def evaluate_rank_scaling(
    n_batches: int,
    batch_size: int,
    input_dim: int = DEFAULT_INPUT_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
) -> None:
    """
    Evaluate how well dendritic layers scale with target rank.

    Tests the ability to learn varying numbers of independent pairwise products.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        n_batches: Number of training batches.
        batch_size: Size of each training batch.
    """
    print("\n" + "-" * 90)
    print("TEST 1: Rank Scaling (how many independent x_i*x_j terms can be learned?)")
    print("-" * 90)

    ranks_to_test = [2, 8, 16, 24]
    poly_ranks_to_test = [4, 8, 16]

    print(f"\n{'Model':<30}", end="")
    for r in ranks_to_test:
        print(f"rank={r:<6}", end="")
    print()
    print("-" * 90)

    for poly_rank in poly_ranks_to_test:
        # Use functools.partial to avoid lambda capture issues
        model_configs = [
            (
                f"DendriticLayer r={poly_rank}",
                functools.partial(
                    DendriticLayer,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    poly_rank=poly_rank,
                    independent_inputs=True,
                ),
            ),
        ]

        for model_name, model_fn in model_configs:
            print(f"{model_name:<30}", end="")

            for target_rank in ranks_to_test:
                # Create target function for this specific rank
                # target_fn = functools.partial(create_rank_k_target, k=target_rank)
                target_fn, _ = make_rank_k_target(
                    k=target_rank, d=input_dim, seed=42, normalize=True
                )

                # Run benchmark
                r2 = run_training_benchmark(
                    model_fn=model_fn,
                    target_fn=target_fn,
                    n_batches=n_batches,
                    batch_size=batch_size,
                    input_dim=input_dim,
                )

                symbol = "✓" if r2 > 0.99 else ("~" if r2 > 0.9 else "✗")
                r2_score_indicator = f"{symbol}{r2:.2f}"
                print(f"{r2_score_indicator:<6}", end="")
            print()
        print()


def test_structure_responsiveness(
    n_batches: int,
    batch_size: int,
    input_dim: int = DEFAULT_INPUT_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    test_poly_rank: int = 8,
) -> None:
    """
    Test sensitivity to different target structures (diagonal vs cross terms).

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        n_batches: Number of training batches.
        batch_size: Size of each training batch.
        test_poly_rank: Polynomial rank to test.
    """
    print("\n" + "-" * 90)
    print("TEST 2: Structure Sensitivity (diagonal x_i² vs cross x_i*x_j)")
    print("-" * 90)

    # Define target functions
    def target_diagonal(X_batch: torch.Tensor) -> torch.Tensor:
        return (X_batch[:, :10] ** 2).sum(dim=1, keepdim=True)

    def target_cross(X_batch: torch.Tensor) -> torch.Tensor:
        terms = [X_batch[:, i] * X_batch[:, i + 1] for i in range(10)]
        return torch.stack(terms, dim=1).sum(dim=1, keepdim=True)

    def target_mixed(X_batch: torch.Tensor) -> torch.Tensor:
        diag = (X_batch[:, :5] ** 2).sum(dim=1)
        cross_terms = [X_batch[:, i] * X_batch[:, i + 5] for i in range(5)]
        cross = torch.stack(cross_terms, dim=1).sum(dim=1)
        return (diag + cross).unsqueeze(1)

    targets_struct = {
        "10×(x_i²)": target_diagonal,
        "10×(x_i·x_{i+1})": target_cross,
        "5×diag + 5×cross": target_mixed,
    }

    # Generate model configurations programmatically
    models_struct: Dict[str, Callable[[], nn.Module]] = {
        "Linear": functools.partial(nn.Linear, input_dim, output_dim),
    }

    # Add DendriticLayer variants
    for poly_rank in [8, 4]:  # , 6, 4, 2]:
        models_struct[f"DendriticLayer r={poly_rank}"] = functools.partial(
            DendriticLayer, input_dim, output_dim, poly_rank=poly_rank
        )

    # Add DendriticStack variants
    for poly_rank in [8, 4]:
        for poly_degree in [3, 4]:
            models_struct[f"DendriticStack r={poly_rank} d={poly_degree}"] = (
                functools.partial(
                    DendriticStack,
                    input_dim,
                    output_dim,
                    poly_rank=poly_rank,
                    poly_degree=poly_degree,
                )
            )

    print(f"\n{'Model':<30}", end="")
    for name in targets_struct.keys():
        print(f"{name:<20}", end="")
    print()
    print("-" * 90)

    for model_name, model_fn in models_struct.items():
        print(f"{model_name:<30}", end="")

        for target_name, target_fn in targets_struct.items():
            r2 = run_training_benchmark(
                model_fn=model_fn,
                target_fn=target_fn,
                n_batches=n_batches,
                batch_size=batch_size,
                input_dim=input_dim,
            )

            symbol = "✓" if r2 > 0.99 else ("~" if r2 > 0.9 else "✗")
            print(f"{symbol}{r2:<19.3f}", end="")
        print()


def assess_cubic_feature_contributions(
    n_batches: int,
    batch_size: int,
    input_dim: int = DEFAULT_INPUT_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    test_poly_rank: int = 8,
) -> None:
    """
    Assess ability to learn higher-order interactions.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        n_batches: Number of training batches.
        batch_size: Size of each training batch.
        test_poly_rank: Polynomial rank to test.
    """
    print("\n" + "-" * 90)
    print("TEST 3: Higher-Order Interactions (when do cubic terms help?)")
    print("-" * 90)

    # Define target functions
    def target_x0_x1(X_batch: torch.Tensor) -> torch.Tensor:
        return (X_batch[:, 0] * X_batch[:, 1]).unsqueeze(1)

    def target_x0_x1_x2(X_batch: torch.Tensor) -> torch.Tensor:
        return (X_batch[:, 0] * X_batch[:, 1] * X_batch[:, 2]).unsqueeze(1)

    def target_x0_x1_x2_x3(X_batch: torch.Tensor) -> torch.Tensor:
        return (
            X_batch[:, 0] * X_batch[:, 1] * X_batch[:, 2] * X_batch[:, 3]
        ).unsqueeze(1)

    def target_sum_product(X_batch: torch.Tensor) -> torch.Tensor:
        return (
            (X_batch[:, 0] + X_batch[:, 1]) * (X_batch[:, 2] + X_batch[:, 3])
        ).unsqueeze(1)

    targets_order = {
        "x₀·x₁": target_x0_x1,
        "x₀·x₁·x₂": target_x0_x1_x2,
        "x₀·x₁·x₂·x₃": target_x0_x1_x2_x3,
        "(x₀+x₁)·(x₂+x₃)": target_sum_product,
    }

    # Generate model configurations programmatically
    models_order: Dict[str, Callable[[], nn.Module]] = {
        "Linear": functools.partial(nn.Linear, input_dim, output_dim),
    }

    # Add DendriticLayer variants
    for poly_rank in [8, 6, 4, 2]:
        models_order[f"DendriticLayer r={poly_rank}"] = functools.partial(
            DendriticLayer, input_dim, output_dim, poly_rank=poly_rank
        )

    # Add DendriticStack variants
    for poly_rank in [8, 6, 4, 2]:
        for poly_degree in [3, 4]:
            models_order[f"DendriticStack r={poly_rank} d={poly_degree}"] = (
                functools.partial(
                    DendriticStack,
                    input_dim,
                    output_dim,
                    poly_rank=poly_rank,
                    poly_degree=poly_degree,
                )
            )

    print(f"\n{'Model':<25}", end="")
    for name in targets_order.keys():
        print(f"{name:<18}", end="")
    print()
    print("-" * 90)

    for model_name, model_fn in models_order.items():
        print(f"{model_name:<25}", end="")

        for target_name, target_fn in targets_order.items():
            r2 = run_training_benchmark(
                model_fn=model_fn,
                target_fn=target_fn,
                n_batches=n_batches,
                batch_size=batch_size,
                input_dim=input_dim,
            )

            symbol = "✓" if r2 > 0.99 else ("~" if r2 > 0.9 else "✗")
            print(f"{symbol}{r2:<17.3f}", end="")
        print()


def diagnostic_benchmark(n_batches: int = 7500, batch_size: int = 16) -> None:
    """
    Diagnostic tests to find actual capability differences.

    Key insight: Test at the EDGE of capacity, not well below it.

    Args:
        n_batches: Number of training batches.
        batch_size: Size of each training batch.
    """
    import time

    torch.manual_seed(42)

    # Smaller dimensions to make rank limits matter
    input_dim = 64
    output_dim = 1
    # Fixed poly_rank for this test
    test_poly_rank = 8

    print("=" * 90)
    print("DIAGNOSTIC BENCHMARK: Finding Architecture Limits")
    print("=" * 90)
    print(
        f"Config: input_dim={input_dim}, batches={n_batches}, batch_size={batch_size}"
    )

    # Test 1: Rank scaling
    evaluate_rank_scaling(n_batches, batch_size, input_dim, output_dim)

    # Test 2: Diagonal vs Cross structure
    test_structure_responsiveness(
        n_batches, batch_size, input_dim, output_dim, test_poly_rank
    )

    # Test 3: Cubic necessity
    assess_cubic_feature_contributions(
        n_batches, batch_size, input_dim, output_dim, test_poly_rank
    )

    # # Test 4: Generalization (train/test split)
    # evaluate_model_generalization(
    #     input_dim,
    #     output_dim,
    #     n_batches,
    #     batch_size,
    # )


def evaluate_model_generalization(
    input_dim: int = DEFAULT_INPUT_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    n_batches: int = 5,
    batch_size: int = 64,
) -> None:
    """
    Evaluate generalization performance with train/test split.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        n_batches: Number of training batches.
        batch_size: Size of each training batch.
    """
    print("\n" + "-" * 90)
    print("TEST 4: Generalization (train on random batches, test on fixed set)")
    print("-" * 90)

    # Fixed test set
    test_size = 1000
    X_test = torch.randn(test_size, input_dim)
    X_test_std = X_test.std()
    if X_test_std == 0:
        X_test_std = EPS
    X_test = X_test / (X_test_std + EPS)
    y_test = create_rank_k_target(X_test, k=16, seed=123)

    # Fixed training evaluation set (not used for training, only for computing train R²)
    train_eval_size = 1000
    X_train_eval = torch.randn(train_eval_size, input_dim)
    X_train_eval_std = X_train_eval.std()
    if X_train_eval_std == 0:
        X_train_eval_std = EPS
    X_train_eval = X_train_eval / (X_train_eval_std + EPS)
    y_train_eval = create_rank_k_target(X_train_eval, k=16, seed=456)

    # Generate model configurations
    models_gen: Dict[str, Callable[[], nn.Module]] = {
        "Linear": functools.partial(nn.Linear, input_dim, output_dim),
    }

    for poly_rank in [8, 6, 4, 2]:
        models_gen[f"DendriticLayer r={poly_rank}"] = functools.partial(
            DendriticLayer, input_dim, output_dim, poly_rank=poly_rank
        )
        models_gen[f"DendriticStack r={poly_rank}"] = functools.partial(
            DendriticStack, input_dim, output_dim, poly_rank=poly_rank
        )

    print(f"\n{'Model':<25} {'Train R²':<12} {'Test R²':<12} {'Gap':<12}")
    print("-" * 60)

    for model_name, model_fn in models_gen.items():
        model = model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop over random batches
        for batch in range(n_batches):
            X_batch = torch.randn(batch_size, input_dim)
            X_batch_std = X_batch.std()
            if X_batch_std == 0:
                X_batch_std = EPS
            X_batch = X_batch / (X_batch_std + EPS)

            y_batch = create_rank_k_target(
                X_batch, k=16, seed=batch
            )  # different seed each batch

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_train = model(X_train_eval)
            pred_test = model(X_test)

        r2_train = (
            1
            - ((pred_train - y_train_eval) ** 2).sum().item()
            / ((y_train_eval - y_train_eval.mean()) ** 2).sum().item()
        )
        r2_test = (
            1
            - ((pred_test - y_test) ** 2).sum().item()
            / ((y_test - y_test.mean()) ** 2).sum().item()
        )

        print(
            f"{model_name:<25} {r2_train:<12.4f} {r2_test:<12.4f} {r2_train - r2_test:<12.4f}"
        )


if __name__ == "__main__":
    diagnostic_benchmark(n_batches=7500, batch_size=16)
