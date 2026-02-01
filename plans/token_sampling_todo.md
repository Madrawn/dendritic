# Todo: Add Token Sampling to Evaluation Process

## Phase 1: Create Sampling Utility

- [ ] **Create `sampling_utils.py`** in `dendritic/experiments/confidence/`
  - Implement `sample_tokens_from_model()` function
  - Handle both `MiniGPT` and `ConfidenceAwareGPT` models
  - Support temperature, top-p sampling
  - Include token decoding logic
  - Add autoregressive sampling for models without generate() method

- [ ] **Add sampling configuration** to `ConfidenceExperimentConfig`
  - Add `sampling_prompt` field with default value
  - Add `sampling_temperature` field (default: 0.8)
  - Add `sampling_top_p` field (default: 0.95)
  - Add `sampling_max_tokens` field (default: 50)

## Phase 2: Modify UnifiedTrainer

- [ ] **Update `UnifiedTrainer.__init__()`**
  - Add tokenizer parameter (optional)
  - Store sampling configuration from config
  - Add `current_seed` tracking for logging
  - Add `results_dir` for saving sampled tokens to file

- [ ] **Add `_sample_and_log()` method** to `UnifiedTrainer`
  - Takes model, step number, and evaluation loss
  - Calls sampling utility
  - Logs sampled tokens with step context (truncated to 100 chars in log)
  - Saves full sampled tokens to text file in results directory
  - Handles errors gracefully

- [ ] **Add `_save_sampled_tokens_to_file()` method**
  - Creates/updates text file with same name as JSON results
  - Appends sampled tokens with timestamp and step info
  - Ensures file is created in correct results directory

- [ ] **Integrate sampling into evaluation flow**
  - Call `_sample_and_log()` after evaluation in training loop (line 380)
  - Call during final evaluation (line 452)
  - Add conditional check for tokenizer availability

## Phase 3: Update Experiment Integration

- [ ] **Modify `ConfidenceAwareExperiment.run()`**
  - Pass tokenizer to UnifiedTrainer constructor
  - Ensure backward compatibility

- [ ] **Update `train_confidence_model()` and `train_standard_model()`**
  - Pass tokenizer when creating UnifiedTrainer

## Phase 4: Testing and Validation

- [ ] **Write unit tests for sampling utilities**
- [ ] **Test integration with existing experiments**
- [ ] **Verify logging output format**
- [ ] **Test backward compatibility**
- [ ] **Test with both model types**

## Implementation Details

### Key Functions to Implement:

1. **`sample_tokens_from_model()`** in `sampling_utils.py`:
   ```python
   def sample_tokens_from_model(model, tokenizer, prompt, max_new_tokens=50, 
                                temperature=0.8, top_p=0.95, device="cuda"):
       # Encode prompt
       # Autoregressive sampling loop
       # Apply temperature and top-p
       # Decode and return
   ```

2. **Autoregressive sampling** (since MiniGPT doesn't have `.generate()`):
   ```python
   def _autoregressive_sample(model, input_ids, max_new_tokens, temperature, top_p):
       for _ in range(max_new_tokens):
           # Get logits for last position
           logits = model(input_ids)
           next_token_logits = logits[:, -1, :] / temperature
           
           # Apply top-p filtering
           sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
           cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
           sorted_indices_to_remove = cumulative_probs > top_p
           
           # Sample and append
           next_token = torch.multinomial(probs, num_samples=1)
           input_ids = torch.cat([input_ids, next_token], dim=-1)
   ```

3. **File saving in UnifiedTrainer**:
   ```python
   def _save_sampled_tokens_to_file(self, step, seed, model_type, eval_loss, sampled_text):
       # Determine results directory from config
       results_dir = Path(self.config.results_dir)
       results_dir.mkdir(parents=True, exist_ok=True)
       
       # Find matching JSON file or create new text file
       # Append sampled tokens with formatting
       with open(text_file_path, "a", encoding="utf-8") as f:
           f.write(f"=== Step {step} (Seed: {seed}, Model: {model_type}) ===\n")
           f.write(f"Timestamp: {datetime.now().isoformat()}\n")
           f.write(f"Eval Loss: {eval_loss:.4f}\n")
           f.write(f"Sampled Tokens ({len(sampled_text.split())}): {sampled_text}\n")
           f.write("=" * 40 + "\n\n")
   ```

### Configuration Changes:

Add to `ConfidenceExperimentConfig` in `config.py`:
```python
sampling_prompt: str = "The quick brown fox jumps over the lazy dog"
sampling_temperature: float = 0.8
sampling_top_p: float = 0.95
sampling_max_tokens: int = 50
```

### Logging Format:

**Console log** (truncated):
```
{model_type} seed={seed} step={step}: eval_loss={loss:.4f}, sampled: {generated_text[:100]}...
```

**Text file format** (full text):
```
=== Step {step} (Seed: {seed}, Model: {model_type}) ===
Timestamp: {timestamp}
Eval Loss: {loss:.4f}
Sampled Tokens (50): {full_generated_text}
========================================
```

## Files to Modify:

1. `dendritic/experiments/confidence/sampling_utils.py` (NEW)
2. `dendritic/experiments/confidence/config.py`
3. `dendritic/experiments/confidence/UnifiedTrainer.py`
4. `dendritic/experiments/confidence/experiment.py`
5. `tests/test_confidence_experiment/test_sampling.py` (NEW tests)

## Dependencies:

- `transformers` for tokenizer (already in project)
- `torch` for tensor operations
- Ensure CUDA/CPU compatibility