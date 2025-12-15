"""
Dendritic computation layers for neural networks.

These layers add learnable polynomial (quadratic) features to standard linear
transformations, inspired by dendritic nonlinear integration in biological neurons.

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
from typing import Optional

from dendritic.layers.DendriticLayer import DendriticLayer
from dendritic.layers.DendriticMLP import DendriticMLP
from dendritic.layers.DendriticStack import DendriticStack



def _test():
    """Quick sanity check."""
    torch.manual_seed(42)

    # Test basic layer
    layer = DendriticLayer(64, 32, poly_rank=8)
    x = torch.randn(16, 64)
    y = layer(x)
    assert y.shape == (16, 32)

    # Test gradient flow
    loss = y.sum()
    loss.backward()
    assert layer.w1.grad is not None
    assert layer.scale.grad is not None

    # Test DendriticStack base case
    stack = DendriticStack(64, 32, poly_rank=8)
    x_stack = torch.randn(16, 64)
    y_stack = stack(x_stack)
    assert y_stack.shape == (16, 32)
    # Test gradient flow for stack
    loss_stack = y_stack.sum()
    loss_stack.backward()
    for w in stack.projections:
        assert w.grad is not None
    assert stack.poly_out.grad is not None
    # Test MLP
    mlp = DendriticMLP(256, 1024, poly_rank=16)
    x = torch.randn(4, 128, 256)  # [batch, seq, embed]
    y = mlp(x)
    assert y.shape == (4, 128, 256)

    # Parameter count
    n_params = sum(p.numel() for p in layer.parameters())
    # Compute expected diag_rank as in __init__
    poly_rank = 8
    if layer.diag_rank is not None and isinstance(layer.diag_rank, int) and layer.diag_rank > 0:
        expected = (64 * 32 + 32) + (2 * poly_rank * 64) + (32 * poly_rank) + 1 + (layer.diag_rank * 64) + (32 * layer.diag_rank) + 1
    else:
        expected = (64 * 32 + 32) + (2 * poly_rank * 64) + (32 * poly_rank) + 1
    assert n_params == expected, f"{n_params} != {expected} (expected calculation may need update)"

    n_params_stack = sum(p.numel() for p in stack.parameters())
    print("All tests passed!")
    print(f"DendriticLayer(64, 32, poly_rank=8): {n_params:,} parameters")
    print(f"DendriticStack(64, 32, poly_rank=8): {n_params_stack:,} parameters")
    print(f"  vs nn.Linear(64, 32): {64*32+32:,} parameters")
    print(f"  overhead: {n_params - (64*32+32):,} ({100*(n_params/(64*32+32)-1):.1f}%)")


def diagnostic_benchmark():
    """
    Diagnostic tests to find actual capability differences.

    Key insight: Test at the EDGE of capacity, not well below it.
    """
    import time

    torch.manual_seed(42)

    # Smaller dimensions to make rank limits matter
    input_dim = 64
    output_dim = 1
    n_samples = 500
    n_epochs = 500

    X = torch.randn(n_samples, input_dim)

    # Normalize inputs to similar scale
    X = X / X.std()

    print("=" * 90)
    print("DIAGNOSTIC BENCHMARK: Finding Architecture Limits")
    print("=" * 90)
    print(f"Config: input_dim={input_dim}, samples={n_samples}, epochs={n_epochs}")

    # Test 1: Rank scaling
    print("\n" + "-" * 90)
    print("TEST 1: Rank Scaling (how many independent x_i*x_j terms can be learned?)")
    print("-" * 90)

    def make_rank_k_target(X, k, seed=42):
        """k independent pairwise products."""
        torch.manual_seed(seed)
        d = X.shape[1]
        # Pick k random pairs (no overlap)
        perm = torch.randperm(d)[: 2 * k]
        pairs = perm.reshape(k, 2)
        result = sum(X[:, i] * X[:, j] for i, j in pairs)
        # Normalize
        result = result / result.std()
        return result.unsqueeze(1)

    ranks_to_test = [2, 4, 8, 12, 16, 24]
    poly_ranks_to_test = [4, 8, 16]

    print(f"\n{'Model':<30}", end="")
    for r in ranks_to_test:
        print(f"rank={r:<6}", end="")
    print()
    print("-" * 90)

    for poly_rank in poly_ranks_to_test:
        model_configs = [
            (
                f"DendriticLayer r={poly_rank}",
                lambda: DendriticLayer(input_dim, output_dim, poly_rank=poly_rank),
            ),
        ]

        for model_name, model_fn in model_configs:
            print(f"{model_name:<30}", end="")

            for target_rank in ranks_to_test:
                y = make_rank_k_target(X, target_rank)

                model = model_fn()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                for epoch in range(n_epochs):
                    optimizer.zero_grad()
                    pred = model(X)
                    loss = F.mse_loss(pred, y)
                    loss.backward()
                    optimizer.step()

                # R² score
                ss_res = ((pred - y) ** 2).sum().item()
                ss_tot = ((y - y.mean()) ** 2).sum().item()
                r2 = 1 - ss_res / ss_tot

                symbol = "✓" if r2 > 0.99 else ("~" if r2 > 0.9 else "✗")
                print(f"{symbol}{r2:<6.2f}", end="")
            print()
        print()

    # Test 2: Diagonal vs Cross structure
    print("\n" + "-" * 90)
    print("TEST 2: Structure Sensitivity (diagonal x_i² vs cross x_i*x_j)")
    print("-" * 90)

    # Fixed poly_rank for this test
    test_poly_rank = 8

    targets_struct = {
        "10×(x_i²)": (X[:, :10] ** 2).sum(dim=1, keepdim=True),
        "10×(x_i·x_{i+1})": sum(X[:, i] * X[:, i + 1] for i in range(10)).unsqueeze(1),
        "5×diag + 5×cross": (
            (X[:, :5] ** 2).sum(dim=1) + sum(X[:, i] * X[:, i + 5] for i in range(5))
        ).unsqueeze(1),
    }

    # Normalize targets
    targets_struct = {k: v / v.std() for k, v in targets_struct.items()}

    models_struct = {
        "Linear": lambda: nn.Linear(input_dim, output_dim),
        f"DendriticLayer r={test_poly_rank}": lambda: DendriticLayer(
            input_dim, output_dim, poly_rank=test_poly_rank
        ),
        f"DendriticLayer r={test_poly_rank//2}": lambda: DendriticLayer(
            input_dim, output_dim, poly_rank=test_poly_rank // 2
        ),
        f"DendriticLayer r={test_poly_rank*2}": lambda: DendriticLayer(
            input_dim, output_dim, poly_rank=test_poly_rank * 2
        ),
        f"DendriticStack r={test_poly_rank}": lambda: DendriticStack(
            input_dim, output_dim, poly_rank=test_poly_rank
        ),
        f"DendriticStack r={test_poly_rank//2}": lambda: DendriticStack(
            input_dim, output_dim, poly_rank=test_poly_rank // 2
        ),
        f"DendriticStack r={test_poly_rank*2}": lambda: DendriticStack(
            input_dim, output_dim, poly_rank=test_poly_rank * 2
        ),

    }

    print(f"\n{'Model':<30}", end="")
    for name in targets_struct.keys():
        print(f"{name:<20}", end="")
    print()
    print("-" * 90)

    for model_name, model_fn in models_struct.items():
        print(f"{model_name:<30}", end="")

        for target_name, y in targets_struct.items():
            model = model_fn()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(n_epochs):
                optimizer.zero_grad()
                pred = model(X)
                loss = F.mse_loss(pred, y)
                loss.backward()
                optimizer.step()

            ss_res = ((pred - y) ** 2).sum().item()
            ss_tot = ((y - y.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot

            symbol = "✓" if r2 > 0.99 else ("~" if r2 > 0.9 else "✗")
            print(f"{symbol}{r2:<19.3f}", end="")
        print()

    # Test 3: Cubic necessity
    print("\n" + "-" * 90)
    print("TEST 3: Higher-Order Interactions (when do cubic terms help?)")
    print("-" * 90)

    targets_order = {
        "x₀·x₁": (X[:, 0] * X[:, 1]).unsqueeze(1),
        "x₀·x₁·x₂": (X[:, 0] * X[:, 1] * X[:, 2]).unsqueeze(1),
        "x₀·x₁·x₂·x₃": (X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]).unsqueeze(1),
        "(x₀+x₁)·(x₂+x₃)": ((X[:, 0] + X[:, 1]) * (X[:, 2] + X[:, 3])).unsqueeze(1),
    }
    targets_order = {k: v / v.std() for k, v in targets_order.items()}

    models_order = {
        "Linear": lambda: nn.Linear(input_dim, output_dim),
        "DendriticLayer r=8": lambda: DendriticLayer(
            input_dim, output_dim, poly_rank=8
        ),
        "DendriticLayer r=6": lambda: DendriticLayer(
            input_dim, output_dim, poly_rank=6
        ),
        "DendriticLayer r=4": lambda: DendriticLayer(
            input_dim, output_dim, poly_rank=4
        ),
        "DendriticLayer r=2": lambda: DendriticLayer(
            input_dim, output_dim, poly_rank=2
        ),
        f"DendriticStack r={test_poly_rank}": lambda: DendriticStack(
            input_dim, output_dim, poly_rank=test_poly_rank
        ),
        f"DendriticStack r={test_poly_rank//2}": lambda: DendriticStack(
            input_dim, output_dim, poly_rank=test_poly_rank // 2
        ),
        f"DendriticStack r={test_poly_rank//4}": lambda: DendriticStack(
            input_dim, output_dim, poly_rank=test_poly_rank // 4
        ),
    }

    print(f"\n{'Model':<25}", end="")
    for name in targets_order.keys():
        print(f"{name:<18}", end="")
    print()
    print("-" * 90)

    for model_name, model_fn in models_order.items():
        print(f"{model_name:<25}", end="")

        for target_name, y in targets_order.items():
            model = model_fn()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(n_epochs):
                optimizer.zero_grad()
                pred = model(X)
                loss = F.mse_loss(pred, y)
                loss.backward()
                optimizer.step()

            ss_res = ((pred - y) ** 2).sum().item()
            ss_tot = ((y - y.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot

            symbol = "✓" if r2 > 0.99 else ("~" if r2 > 0.9 else "✗")
            print(f"{symbol}{r2:<17.3f}", end="")
        print()

    # Test 4: Generalization (train/test split)
    print("\n" + "-" * 90)
    print("TEST 4: Generalization (train on 80%, test on 20%)")
    print("-" * 90)

    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]

    # Medium-complexity target
    y_full = make_rank_k_target(X, k=8, seed=123)
    y_train, y_test = y_full[:n_train], y_full[n_train:]

    models_gen = {
        "Linear": lambda: nn.Linear(input_dim, output_dim),
        "DendriticLayer r=8": lambda: DendriticLayer(
            input_dim, output_dim, poly_rank=8
        ),
        "DendriticLayer r=6": lambda: DendriticLayer(
            input_dim, output_dim, poly_rank=6
        ),
        "DendriticLayer r=4": lambda: DendriticLayer(
            input_dim, output_dim, poly_rank=4
        ),
        "DendriticLayer r=2": lambda: DendriticLayer(
            input_dim, output_dim, poly_rank=2
        ),
        f"DendriticStack r={test_poly_rank}": lambda: DendriticStack(
            input_dim, output_dim, poly_rank=test_poly_rank
        ),
        f"DendriticStack r={test_poly_rank//2}": lambda: DendriticStack(
            input_dim, output_dim, poly_rank=test_poly_rank // 2
        ),
        f"DendriticStack r={test_poly_rank//4}": lambda: DendriticStack(
            input_dim, output_dim, poly_rank=test_poly_rank // 4
        ),
    }

    print(f"\n{'Model':<25} {'Train R²':<12} {'Test R²':<12} {'Gap':<12}")
    print("-" * 60)

    for model_name, model_fn in models_gen.items():
        model = model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            pred = model(X_train)
            loss = F.mse_loss(pred, y_train)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_train = model(X_train)
            pred_test = model(X_test)

        r2_train = (
            1
            - ((pred_train - y_train) ** 2).sum().item()
            / ((y_train - y_train.mean()) ** 2).sum().item()
        )
        r2_test = (
            1
            - ((pred_test - y_test) ** 2).sum().item()
            / ((y_test - y_test.mean()) ** 2).sum().item()
        )

        print(
            f"{model_name:<25} {r2_train:<12.4f} {r2_test:<12.4f} {r2_train - r2_test:<12.4f}"
        )


def check_true_capacity():
    torch.manual_seed(42)
    # Increase samples to prevent memorization
    N = 10000 
    d = 20 # Smaller dim to ensure density
    
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
        "DendriticLayer": DendriticLayer(d, 1, poly_rank=16), # Pure Quadratic
        "DendriticStack": DendriticStack(d, 1, poly_rank=16), # Quadratic + GELU + Quadratic
    }
    
    for name, model in models.items():
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        # Train
        for _ in range(500):
            opt.zero_grad()
            loss = F.mse_loss(model(X_train), y_train)
            loss.backward()
            opt.step()
            
        # Eval
        with torch.no_grad():
            r2_train = 1 - F.mse_loss(model(X_train), y_train) / y_train.var()
            r2_test = 1 - F.mse_loss(model(X_test), y_test) / y_test.var()
            
        print(f"{name:<20} {r2_train.item():<10.3f} {r2_test.item():<10.3f}")
        
def performance_test():
    """Test all implementations for correctness and expressiveness."""
    import time
    
    torch.manual_seed(42)
    
    batch_size = 64
    input_dim = 256
    output_dim = 128
    num_iterations = 500
    
    print("=" * 80)
    print("Dendritic Layer Comprehensive Benchmark")
    print("=" * 80)
    print(f"\nConfig: batch={batch_size}, in={input_dim}, out={output_dim}, iters={num_iterations}")
    
    # Test data
    x = torch.randn(batch_size, input_dim)
    
    # Models to test
    models = {
        "Linear (baseline)": nn.Linear(input_dim, output_dim),
        "Dendritic (rank=8)": DendriticLayer(input_dim, output_dim, poly_rank=8),
        "Dendritic (rank=16)": DendriticLayer(input_dim, output_dim, poly_rank=16),
        "DendriticV2 (rank=8)": DendriticLayer(input_dim, output_dim, poly_rank=8),
        "DendriticV2 (rank=16)": DendriticLayer(input_dim, output_dim, poly_rank=16),        
        "DendriticStack (rank=8)": DendriticStack(input_dim, output_dim, poly_rank=8),
        "DendriticStack (rank=16)": DendriticStack(input_dim, output_dim, poly_rank=16),
    }
    
    # Add two-layer versions for fair comparison
    def make_two_layer(cls, **kwargs):
        return nn.Sequential(
            cls(input_dim, output_dim, **kwargs),
            nn.GELU(),
            cls(output_dim, output_dim, **kwargs)
        )
    
    # Warmup
    print("\nWarming up...")
    for model in models.values():
        for _ in range(50):
            _ = model(x)
    
    # Timing benchmark
    print("\n" + "-" * 80)
    print("TIMING BENCHMARK")
    print("-" * 80)
    
    baseline_time = None
    results = {}
    
    for name, model in models.items():
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = model(x)
        elapsed = time.perf_counter() - start
        
        if baseline_time is None:
            baseline_time = elapsed
        
        num_params = sum(p.numel() for p in model.parameters())
        baseline_params = 256 * 128 + 128  # Linear baseline
        
        results[name] = {
            "time_ms": elapsed * 1000 / num_iterations,
            "slowdown": elapsed / baseline_time,
            "params": num_params,
            "extra_params": num_params - baseline_params,
        }
    
    print(f"\n{'Model':<25} {'Time(ms)':<10} {'Slowdown':<10} {'Params':<12} {'Extra':<10}")
    print("-" * 80)
    for name, res in results.items():
        print(f"{name:<25} {res['time_ms']:<10.3f} {res['slowdown']:<10.2f}x "
              f"{res['params']:<12,} {res['extra_params']:<+10,}")
    
        
        

if __name__ == "__main__":
    _test()
    check_true_capacity()
    performance_test()
    # diagnostic_benchmark()

