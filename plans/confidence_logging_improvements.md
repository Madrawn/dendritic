# Confidence Model Logging Improvements Plan

## Objective
Improve logging for confidence-aware models to show language modeling loss (lm_loss) and confidence loss separately instead of combined total loss.

## Key Insights from Code Analysis
1. **Training**: `two_pass_training_step` returns `total_loss`, `loss_lm`, `loss_confidence`
2. **Evaluation**: Currently returns only `loss` (which is LM loss, not confidence loss)
3. **Progress bar**: Shows `total_loss` as "loss"
4. **Logging**: Shows `avg_train_loss` (total loss) and `avg_eval_loss` (LM loss)

## Requirements from User
1. Modify logging line to show both losses separately
2. Update progress bar to show lm_loss instead of total loss
3. Keep evaluation loss as LM loss (already correct)

## Implementation Strategy

### Approach: Show LM Loss as Primary Metric
Since evaluation only computes LM loss, we should:
1. Show `lm_loss` as the primary "loss" in progress bar
2. Show both `lm_loss` and `conf_loss` in logging
3. Keep `eval_loss` as LM loss (already correct)

### Phase 1: Update Training Loss Tracking
1. **Track separate moving averages** for `lm_loss` and `conf_loss`
2. **Update progress bar** to show `lm_loss` instead of `total_loss`
3. **Modify logging format** to show both losses

### Phase 2: Handle Model Type Detection
1. **Detect confidence model** vs standard model
2. **Conditional formatting** based on model type
3. **Fallback** for standard models (show only `loss`)

## Detailed Implementation Steps

### Step 1: Analyze Current Loss Tracking
- `avg_train_loss_tensor` tracks `total_loss`
- Need separate tensors for `lm_loss` and `conf_loss`

### Step 2: Update UnifiedTrainer.__init__
- Add `avg_train_lm_loss_tensor` and `avg_train_conf_loss_tensor`
- Initialize appropriately

### Step 3: Update Training Loop
- Extract `loss_lm` and `loss_confidence` from step_result
- Update separate moving averages
- Use `loss_lm` for progress bar display

### Step 4: Update Progress Bar Display
- For confidence models: show `lm_loss` as "loss"
- Optionally add `conf_loss` to postfix
- For standard models: keep current behavior

### Step 5: Update Logging Format
- New format: `train_lm=3.4447, train_conf=0.1234, avg_eval_loss=3.2839`
- For standard models: `train=3.4447, avg_eval_loss=3.2839`

### Step 6: Update Evaluation Logic
- Already uses `loss_lm` if available (line 117 in UnifiedTrainer)
- No changes needed for evaluation

## Files to Modify
1. `dendritic/experiments/confidence/UnifiedTrainer.py` - Primary changes
2. `dendritic/experiments/confidence/ConfidenceTrainingStrategy.py` - Minor updates if needed

## Implementation Details

### Changes to UnifiedTrainer:
1. **Add separate loss tracking**:
   ```python
   avg_train_lm_loss_tensor = torch.tensor(15.0, device="cpu")
   avg_train_conf_loss_tensor = torch.tensor(0.0, device="cpu")
   ```

2. **Update loss accumulation**:
   ```python
   if "loss_lm" in step_result:
       avg_train_lm_loss_tensor = avg_train_lm_loss_tensor * 0.9 + 0.1 * step_result["loss_lm"].detach().cpu()
   if "loss_confidence" in step_result:
       avg_train_conf_loss_tensor = avg_train_conf_loss_tensor * 0.9 + 0.1 * step_result["loss_confidence"].detach().cpu()
   ```

3. **Update progress bar**:
   ```python
   # Use lm_loss for display if available
   display_loss = step_result.get("loss_lm", loss)
   ```

4. **Update logging**:
   ```python
   if "loss_confidence" in step_result:
       # Confidence model format
       logging.info(f"{self.model_type} seed={seed} step={step+1}: "
                    f"train_lm={avg_train_lm_loss:.4f}, train_conf={avg_train_conf_loss:.4f}, "
                    f"avg_eval_loss={avg_eval_loss:.4f}, ppl={perplexity:.2f}")
   else:
       # Standard model format
       logging.info(f"{self.model_type} seed={seed} step={step+1}: "
                    f"train={avg_train_loss:.4f}, avg_eval_loss={avg_eval_loss:.4f}, ppl={perplexity:.2f}")
   ```

## Testing Strategy
1. Test with confidence model training
2. Test with standard model training
3. Verify logging output matches expected formats
4. Ensure no regression in training performance

## Success Criteria
1. Logging shows separate lm_loss and conf_loss for confidence models
2. Progress bar shows lm_loss for confidence models
3. Standard models continue to work with existing format
4. Evaluation metrics remain correct
5. No performance degradation