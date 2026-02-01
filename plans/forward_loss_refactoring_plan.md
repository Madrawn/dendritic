# Forward Loss Refactoring Plan

## Problem Statement
The current `MiniGPT` and `ConfidenceAwareGPT` forward methods return loss when labels are provided, violating PyTorch conventions and creating architectural impurity. This makes the models less reusable and creates inconsistent patterns across the codebase.

## Current Architecture Analysis

### Current Forward Method Patterns

1. **MiniGPT.forward()** (lines 92-105):
   - Returns `{"logits": logits}` when no labels
   - Returns `{"logits": logits, "loss": loss}` when labels provided
   - Uses internal `compute_logits_and_loss()` method

2. **ConfidenceAwareGPT.forward()** (lines 182-226):
   - Returns `{"logits": logits, "confidence_pred": confidence_pred}` when no labels
   - Returns same dict plus `"loss": loss` when labels provided
   - Note: Does NOT compute confidence loss in forward (requires two-pass lookahead)

3. **two_pass_training_step()** (lines 242-316):
   - Static method that computes loss externally using `F.cross_entropy`
   - Doesn't use the loss from forward method
   - Creates duplication of loss computation logic

### Current Dependencies
The loss from forward is used in:
- `StandardTrainingStrategy.training_step()` and `.evaluation_step()`
- `ConfidenceTrainingStrategy.evaluation_step()`
- `dendritic/experiments/utils/experiment_pretraining.py`
- `dendritic/experiments/confidence/experiment.py`
- `dendritic/experiments/analysis/evaluation.py`

## Proposed Solution

### Phase 1: Separate Loss Computation from Forward
1. Modify `MiniGPT.forward()` to always return only model outputs (logits)
2. Move loss computation to separate methods or external training logic
3. Update all callers to compute loss externally

### Phase 2: Create Consistent Loss Computation Utilities
1. Create a `loss_utils.py` module with standardized loss functions
2. Ensure both standard and confidence-aware training use the same patterns
3. Update `two_pass_training_step` to use the new utilities

### Phase 3: Update Training Strategies
1. Modify `StandardTrainingStrategy` to compute loss externally
2. Modify `ConfidenceTrainingStrategy` to compute loss externally
3. Ensure backward compatibility with existing experiment code

## Detailed Implementation Plan

### 1. Create Loss Computation Utilities
```python
# dendritic/experiments/utils/loss_utils.py
def compute_language_modeling_loss(logits, labels, ignore_index=-100):
    """Compute standard language modeling loss."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

def compute_confidence_loss(confidence_pred, future_losses, alpha=1.0):
    """Compute confidence prediction loss."""
    return F.mse_loss(confidence_pred, future_losses.detach())
```

### 2. Refactor MiniGPT Class
- Remove `compute_logits_and_loss()` method
- Remove `calculate_loss()` method (move to loss_utils)
- Update `forward()` to return only logits (no loss)
- Add `get_logits()` helper method if needed

### 3. Refactor ConfidenceAwareGPT Class
- Update `forward()` to return only `{"logits": logits, "confidence_pred": confidence_pred}`
- Remove loss computation from forward
- Update `two_pass_training_step()` to use new loss utilities

### 4. Update Training Strategies
- `StandardTrainingStrategy`: Compute loss using `loss_utils.compute_language_modeling_loss()`
- `ConfidenceTrainingStrategy`: Update evaluation step to compute loss externally

### 5. Update Experiment Code
- Update `experiment_pretraining.py` to compute loss externally
- Update `experiment.py` to compute loss externally
- Update `evaluation.py` to compute loss externally

## Benefits
1. **Architectural purity**: Follows PyTorch conventions
2. **Better reusability**: Models can be used without training assumptions
3. **Consistency**: Single source of truth for loss computation
4. **Testability**: Loss functions can be tested independently
5. **Flexibility**: Easier to experiment with different loss functions

## Risks and Mitigations
1. **Breaking changes**: Need to update all callers simultaneously
   - Mitigation: Create comprehensive test coverage before refactoring
2. **Performance impact**: Minimal - loss computation is the same, just moved
3. **Code duplication**: Ensure loss utilities are used consistently

## Testing Strategy
1. Create unit tests for new loss utilities
2. Update existing model tests to verify forward returns only outputs
3. Update integration tests to use new loss computation pattern
4. Run full test suite to ensure no regressions

## Migration Path
1. Implement loss utilities first
2. Update one component at a time (e.g., start with StandardTrainingStrategy)
3. Run tests after each component update
4. Update documentation and examples

## Success Criteria
1. All tests pass
2. Forward methods return only model outputs (no loss)
3. Loss computation is consistent across the codebase
4. Training performance remains unchanged
5. Code is more maintainable and follows PyTorch conventions