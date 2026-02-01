# Confidence Logging Improvements - Implementation Todo

## Overview
Update confidence model logging to show lm_loss and confidence_loss separately instead of combined total loss.

## Files to Modify
1. `dendritic/experiments/confidence/UnifiedTrainer.py`
2. (Optional) `dendritic/experiments/confidence/ConfidenceTrainingStrategy.py`

## Implementation Steps

### 1. Add Separate Loss Tracking Variables
- Add `avg_train_lm_loss_tensor` and `avg_train_conf_loss_tensor` to `__init__` or training state
- Initialize with appropriate values (15.0 for lm_loss, 0.0 for conf_loss)

### 2. Update Loss Accumulation in Training Loop
- Extract `loss_lm` and `loss_confidence` from `step_result`
- Update separate moving averages for each loss component
- Keep `avg_train_loss_tensor` for backward compatibility if needed

### 3. Update Progress Bar Display
- Determine which loss to show as "loss" in progress bar
  - For confidence models: use `loss_lm`
  - For standard models: use `total_loss`
- Update `progress.set_postfix()` to show appropriate loss

### 4. Update Logging Format
- Modify the logging.info() call at line 394
- For confidence models: format as `train_lm=X.XXXX, train_conf=X.XXXX, avg_eval_loss=X.XXXX`
- For standard models: keep existing format `train=X.XXXX, avg_eval_loss=X.XXXX`

### 5. Update History Entry
- Ensure `loss_history` entries include both losses for confidence models
- Already includes `train_loss_lm` and `train_loss_conf` (lines 385-390)

### 6. Handle Edge Cases
- Standard models (no `loss_confidence` in step_result)
- Early training steps where losses might not be available
- Evaluation logic (already handles `loss_lm` vs `loss`)

## Code Changes Preview

### In UnifiedTrainer.train() method:
```python
# Add to training state initialization (around line 270)
avg_train_lm_loss_tensor = torch.tensor(15.0, device="cpu")
avg_train_conf_loss_tensor = torch.tensor(0.0, device="cpu")

# In training loop (after step_result is obtained)
if "loss_lm" in step_result:
    avg_train_lm_loss_tensor = avg_train_lm_loss_tensor * 0.9 + 0.1 * step_result["loss_lm"].detach().cpu()
if "loss_confidence" in step_result:
    avg_train_conf_loss_tensor = avg_train_conf_loss_tensor * 0.9 + 0.1 * step_result["loss_confidence"].detach().cpu()

# Update progress bar (around line 335)
display_loss = step_result.get("loss_lm", loss)
if queued_loss is not None:
    progress.set_postfix({
        "loss": f"{display_loss.item():.4f}",
        "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
    })

# Update logging (around line 394)
if "loss_confidence" in step_result:
    avg_train_lm_loss = avg_train_lm_loss_tensor.item()
    avg_train_conf_loss = avg_train_conf_loss_tensor.item()
    logging.info(
        f"{self.model_type} seed={seed} step={step+1}: "
        f"train_lm={avg_train_lm_loss:.4f}, train_conf={avg_train_conf_loss:.4f}, "
        f"avg_eval_loss={avg_eval_loss:.4f}, ppl={perplexity:.2f}"
    )
else:
    logging.info(
        f"{self.model_type} seed={seed} step={step+1}: "
        f"train={avg_train_loss:.4f}, avg_eval_loss={avg_eval_loss:.4f}, ppl={perplexity:.2f}"
    )
```

## Testing Checklist
- [ ] Run confidence model training - verify new logging format
- [ ] Run standard model training - verify unchanged logging format
- [ ] Check progress bar shows correct loss values
- [ ] Verify loss_history contains both losses for confidence models
- [ ] Ensure evaluation metrics remain correct