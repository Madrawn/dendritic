# Parameter Count Refactoring Plan

## Problem Statement
Parameter counting logic for `DendriticLayer` and `DendriticStack` is currently duplicated in `param_utils.py`. When the layer implementation changes, the parameter counting functions may become outdated, causing test failures. The goal is to move the parameter counting logic into the classes themselves, making it obvious that updates are needed when the class changes.

## Proposed Solution
Add class methods to each dendritic class that calculate parameter count based on initialization parameters. Keep the existing `param_utils` functions as wrappers for backward compatibility.

### Step 1: Add Helper Method for Diagonal Rank Calculation
Create a static method `_compute_diag_rank` in both `DendriticLayer` and `DendriticStack` (or a common base class) that replicates the auto‑logic:
```python
@staticmethod
def _compute_diag_rank(diag_rank, independent_inputs, poly_rank):
    if diag_rank is None or diag_rank == "auto":
        return poly_rank if independent_inputs else max(4, poly_rank // 4)
    return diag_rank
```

### Step 2: Add Class Method `parameter_count`
Add a `@classmethod` to each class with the same signature as `__init__` (excluding `self`). The method returns the total number of parameters (including bias, scale, diagonal pathway, etc.).

**For DendriticLayer:**
```python
@classmethod
def parameter_count(cls, input_dim, output_dim, poly_rank, independent_inputs=False, diag_rank="auto", init_scale=0.1, bias=True):
    # compute effective diag_rank
    effective_diag_rank = cls._compute_diag_rank(diag_rank, independent_inputs, poly_rank)
    # count linear, cross‑term, diagonal parameters
    ...
```

**For DendriticStack:**
```python
@classmethod
def parameter_count(cls, input_dim, output_dim, poly_rank, poly_degree=3, independent_inputs=False, diag_rank="auto", init_scale=0.1, bias=True):
    ...
```

### Step 3: Update `param_utils.py`
Modify `count_dendritic_layer_params` and `count_dendritic_stack_params` to delegate to the respective class methods. This ensures a single source of truth.

Example:
```python
def count_dendritic_layer_params(input_dim, output_dim, poly_rank, diag_rank, bias=True):
    from dendritic.layers.DendriticLayer import DendriticLayer
    # diag_rank must be int (not "auto") – callers already handle this.
    return DendriticLayer.parameter_count(
        input_dim=input_dim,
        output_dim=output_dim,
        poly_rank=poly_rank,
        independent_inputs=False,  # default
        diag_rank=diag_rank,
        bias=bias
    )
```

### Step 4: Update Composite Functions
Update `calculate_mlp_params_dendritic` and `calculate_mlp_params_dendritic_stack` to use the updated counting functions (they already do). No change needed if step 3 is done correctly.

### Step 5: Add Validation Tests
Create or extend existing tests to verify that the class method matches the actual parameter count of instantiated layers across a range of hyperparameters.

Add a test in `tests/test_param_count_consistency.py` (or a new test file) that:
- Randomly samples reasonable hyperparameters.
- Instantiates the layer.
- Compares `sum(p.numel() for p in layer.parameters())` with `layer.parameter_count(...)`.

### Step 6: Run Full Test Suite
Execute `python run_tests.py --mode unit` to ensure no regressions.

## Considerations
- **Backward Compatibility**: The existing function signatures in `param_utils` must remain unchanged.
- **Performance**: The class methods are called during experiment setup, not during training, so performance is not critical.
- **Extensibility**: If new parameters are added to the layers, the `parameter_count` method must be updated accordingly – but that is exactly the desired behavior.

## Deliverables
1. Updated `DendriticLayer.py` with `_compute_diag_rank` and `parameter_count`.
2. Updated `DendriticStack.py` with `_compute_diag_rank` and `parameter_count`.
3. Updated `param_utils.py` with delegated implementations.
4. Additional unit tests for validation.
5. All existing tests pass.

## Next Steps
Once the plan is approved, switch to **Code** mode to implement the changes.