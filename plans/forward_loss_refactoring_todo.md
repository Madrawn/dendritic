# Forward Loss Refactoring - Todo List

## Phase 1: Create Loss Utilities
- [ ] Create `dendritic/experiments/utils/loss_utils.py` with:
  - `compute_language_modeling_loss()` function
  - `compute_confidence_loss()` function
  - Import `torch.nn.functional as F`
- [ ] Add unit tests for loss utilities in `tests/test_loss_utils.py`
- [ ] Update imports in relevant files

## Phase 2: Refactor MiniGPT Base Class
- [ ] Remove `calculate_loss()` method from `MiniGPT` class
- [ ] Remove `compute_logits_and_loss()` method from `MiniGPT` class
- [ ] Update `MiniGPT.forward()` to return only logits (not dict with loss)
- [ ] Add `get_logits()` helper method if needed for consistency
- [ ] Update `MiniGPT.__init__` documentation
- [ ] Update tests for `MiniGPT` to verify forward returns only logits

## Phase 3: Refactor ConfidenceAwareGPT Class
- [ ] Update `ConfidenceAwareGPT.forward()` to return dict without loss
- [ ] Modify `two_pass_training_step()` to use `loss_utils` functions
- [ ] Remove loss computation from forward method
- [ ] Update class documentation
- [ ] Update tests for `ConfidenceAwareGPT`

## Phase 4: Update StandardTrainingStrategy
- [ ] Modify `training_step()` to compute loss using `loss_utils.compute_language_modeling_loss()`
- [ ] Modify `evaluation_step()` to compute loss using `loss_utils.compute_language_modeling_loss()`
- [ ] Update imports to include `loss_utils`
- [ ] Update tests for `StandardTrainingStrategy`

## Phase 5: Update ConfidenceTrainingStrategy
- [ ] Modify `evaluation_step()` to compute loss using `loss_utils.compute_language_modeling_loss()`
- [ ] Ensure `training_step()` still works with updated `two_pass_training_step()`
- [ ] Update imports
- [ ] Update tests for `ConfidenceTrainingStrategy`

## Phase 6: Update Experiment Code
- [ ] Update `dendritic/experiments/utils/experiment_pretraining.py`:
  - Replace `outputs["loss"]` with `loss_utils.compute_language_modeling_loss()`
  - Update forward calls to not expect loss in output
- [ ] Update `dendritic/experiments/confidence/experiment.py`:
  - Replace `outputs["loss"]` with `loss_utils.compute_language_modeling_loss()`
- [ ] Update `dendritic/experiments/analysis/evaluation.py`:
  - Replace `outputs["loss"]` with `loss_utils.compute_language_modeling_loss()`

## Phase 7: Update Other Dependencies
- [ ] Search for any other files using `outputs["loss"]` or `.forward()` with labels
- [ ] Update those files to use new pattern
- [ ] Check `dendritic/experiments/run_experiments.py` for any dependencies
- [ ] Update any example scripts or notebooks

## Phase 8: Testing and Validation
- [ ] Run full test suite: `python run_tests.py --mode all`
- [ ] Run confidence experiment tests: `python run_tests.py -k "confidence"`
- [ ] Run integration tests: `python run_tests.py --mode integration`
- [ ] Verify training still works with a small smoke test
- [ ] Check for any performance regressions

## Phase 9: Documentation and Cleanup
- [ ] Update `AGENTS.md` or other documentation with new patterns
- [ ] Update function docstrings to reflect changes
- [ ] Remove any unused imports or code
- [ ] Create migration guide if needed

## Implementation Notes

### Key Changes to Forward Methods:
```python
# Before:
def forward(self, input_ids, labels=None):
    # ... computation ...
    if labels is not None:
        loss = self.calculate_loss(labels, logits)
        return {"logits": logits, "loss": loss}
    return {"logits": logits}

# After:
def forward(self, input_ids):
    # ... computation ...
    return logits  # or {"logits": logits, "confidence_pred": ...} for ConfidenceAwareGPT
```

### Loss Computation Pattern:
```python
# Before:
outputs = model(input_ids, labels=labels)
loss = outputs["loss"]

# After:
logits = model(input_ids)  # or outputs = model(input_ids)
loss = compute_language_modeling_loss(logits, labels)
```

### ConfidenceAwareGPT Special Case:
```python
# Before:
outputs = model(input_ids, labels=labels, confidence_scalars=conf)
loss = outputs["loss"]  # Only LM loss, not confidence loss

# After:
outputs = model(input_ids, confidence_scalars=conf)
loss_lm = compute_language_modeling_loss(outputs["logits"], labels)
# Confidence loss computed separately in two_pass_training_step
```

## Risk Mitigation
1. **Backward Compatibility**: Consider keeping old methods with deprecation warnings
2. **Incremental Changes**: Update one file at a time, test after each
3. **Version Control**: Use feature branch for all changes
4. **Rollback Plan**: Have backup of original files

## Success Metrics
- All existing tests pass
- No change in training behavior or results
- Code follows PyTorch conventions
- Loss computation is centralized and consistent
- Models are more reusable outside training context