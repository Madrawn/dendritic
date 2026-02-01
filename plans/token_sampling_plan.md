# Plan: Add Token Sampling to Evaluation Process

## Current State Analysis

The `UnifiedTrainer.py` file contains evaluation logic in two main methods:
1. `_evaluate_with_iterator()` - Used during training evaluations (lines 89-127)
2. `_evaluate()` - Alternative evaluation method (lines 129-169)

Evaluation happens during training at regular intervals (every `eval_interval` steps) in the main training loop (lines 367-444).

## Requirements

Add functionality to sample 50 tokens from the current model during evaluation and print the result.

## Design Decisions

Based on user feedback:
1. **When to sample**: During each evaluation step (every `eval_interval`)
2. **Sampling source**: Use a fixed prompt from config
3. **Implementation**: Create separate sampling utility and call it from UnifiedTrainer evaluation
4. **Output**: 
   - Print truncated sampled tokens (first 100 chars) to log
   - Save full sampled tokens to text file alongside JSON results file

## Implementation Plan

### Phase 1: Create Sampling Utility

1. **Create `sampling_utils.py`** in `dendritic/experiments/confidence/`:
   - Implement `sample_tokens_from_model()` function
   - Handle both `MiniGPT` and `ConfidenceAwareGPT` models
   - Support temperature, top-p, and other sampling parameters
   - Include token decoding using appropriate tokenizer

2. **Add sampling configuration** to `ConfidenceExperimentConfig`:
   - Add `sampling_prompt` field with default value
   - Add `sampling_temperature` field (default: 0.8)
   - Add `sampling_top_p` field (default: 0.95)
   - Add `sampling_max_tokens` field (default: 50)

### Phase 2: Integrate Sampling into UnifiedTrainer

3. **Modify `UnifiedTrainer.__init__()`**:
   - Store tokenizer reference (needs to be passed from experiment)
   - Or store sampling configuration

4. **Add `_sample_and_log()` method** to `UnifiedTrainer`:
   - Takes model, step number, and evaluation loss as parameters
   - Calls sampling utility
   - Logs sampled tokens with step context

5. **Integrate sampling into evaluation flow**:
   - Call `_sample_and_log()` after evaluation in training loop (line 380)
   - Also call during final evaluation (line 452)

### Phase 3: Update Training Strategy Interface (Optional)

6. **Consider adding sampling method to `TrainingStrategy`**:
   - Optional enhancement for strategy-specific sampling
   - Could be added later if needed

### Phase 4: Update Experiment Configuration

7. **Modify `ConfidenceExperimentConfig`**:
   - Add sampling-related fields
   - Ensure backward compatibility

8. **Update experiment setup**:
   - Pass tokenizer to UnifiedTrainer
   - Or pass sampling configuration

## Detailed Implementation Steps

### Step 1: Create sampling_utils.py

```python
"""Sampling utilities for model evaluation."""

import torch
import torch.nn as nn
from typing import Optional

def sample_tokens_from_model(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: str = "cuda",
) -> str:
    """
    Sample tokens from a language model.
    
    Args:
        model: The model to sample from
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Device to run on
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate tokens
    with torch.no_grad():
        # For MiniGPT models (no built-in generate method)
        # We need to implement autoregressive sampling
        generated = _autoregressive_sample(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            tokenizer=tokenizer,
        )
    
    return generated

def _autoregressive_sample(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tokenizer,
) -> str:
    """Autoregressive sampling for models without generate() method."""
    # Implementation depends on model type
    # This is a placeholder - actual implementation needed
    pass
```

### Step 2: Update ConfidenceExperimentConfig

Add to `config.py`:

```python
@dataclass
class ConfidenceExperimentConfig(PretrainingConfig):
    # ... existing fields ...
    
    # Sampling configuration
    sampling_prompt: str = "The quick brown fox jumps over the lazy dog"
    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.95
    sampling_max_tokens: int = 50
```

### Step 3: Modify UnifiedTrainer

Add to `UnifiedTrainer.__init__()`:

```python
def __init__(
    self,
    config: ConfidenceExperimentConfig,
    strategy: TrainingStrategy,
    model_type: str,
    tokenizer=None,  # Optional tokenizer for sampling
    results_dir=None,  # Directory for saving sampled tokens
):
    # ... existing initialization ...
    self.tokenizer = tokenizer
    self.results_dir = Path(results_dir) if results_dir else Path(config.results_dir)
    self.sampling_config = {
        "prompt": config.sampling_prompt,
        "temperature": config.sampling_temperature,
        "top_p": config.sampling_top_p,
        "max_tokens": config.sampling_max_tokens,
    }
```

Add `_sample_and_log()` method:

```python
def _sample_and_log(self, model: nn.Module, step: int, eval_loss: float):
    """Sample tokens from model and log results."""
    if self.tokenizer is None:
        logging.warning("No tokenizer provided, skipping sampling")
        return
    
    try:
        from dendritic.experiments.confidence.sampling_utils import sample_tokens_from_model
        
        generated = sample_tokens_from_model(
            model=model,
            tokenizer=self.tokenizer,
            prompt=self.sampling_config["prompt"],
            max_new_tokens=self.sampling_config["max_tokens"],
            temperature=self.sampling_config["temperature"],
            top_p=self.sampling_config["top_p"],
            device=self.device,
        )
        
        # Log truncated version to console
        truncated = generated[:100] + "..." if len(generated) > 100 else generated
        logging.info(
            f"{self.model_type} seed={self.current_seed} step={step}: "
            f"eval_loss={eval_loss:.4f}, sampled: {truncated}"
        )
        
        # Save full version to text file
        self._save_sampled_tokens_to_file(
            step=step,
            seed=self.current_seed,
            model_type=self.model_type,
            eval_loss=eval_loss,
            sampled_text=generated,
        )
        
    except Exception as e:
        logging.warning(f"Sampling failed: {e}")
```

Add `_save_sampled_tokens_to_file()` method:

```python
def _save_sampled_tokens_to_file(self, step, seed, model_type, eval_loss, sampled_text):
    """Save sampled tokens to text file alongside JSON results."""
    from datetime import datetime
    from pathlib import Path
    
    # Ensure results directory exists
    self.results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create text file name based on timestamp or match existing JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_file = self.results_dir / f"{timestamp}_sampled_tokens.txt"
    
    # Append to file
    with open(text_file, "a", encoding="utf-8") as f:
        f.write(f"=== Step {step} (Seed: {seed}, Model: {model_type}) ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Eval Loss: {eval_loss:.4f}\n")
        f.write(f"Sampled Tokens: {sampled_text}\n")
        f.write("=" * 40 + "\n\n")
```

### Step 4: Integrate into Training Loop

In `train()` method, after evaluation (around line 380):

```python
# After evaluation
eval_loss = self._evaluate_with_iterator(...)

# Sample and log
self._sample_and_log(model, step + 1, eval_loss)
```

In final evaluation (around line 452):

```python
# Final evaluation
final_eval_loss = self._evaluate_with_iterator(...)

# Sample and log final state
self._sample_and_log(model, training_steps, final_eval_loss)
```

## Considerations

1. **Tokenizer Availability**: Need to ensure tokenizer is passed from experiment to trainer
2. **Model Compatibility**: Different sampling logic for `MiniGPT` vs `ConfidenceAwareGPT`
3. **Performance Impact**: Sampling adds overhead to evaluation
4. **Logging Volume**: 50 tokens per evaluation could be verbose
5. **Backward Compatibility**: Changes should not break existing experiments

## Testing Plan

1. Unit tests for sampling utilities
2. Integration test with actual model
3. Verify logging output format
4. Test with both model types
5. Test configuration defaults

## Files to Modify

1. `dendritic/experiments/confidence/UnifiedTrainer.py`
2. `dendritic/experiments/confidence/config.py`
3. New file: `dendritic/experiments/confidence/sampling_utils.py`
4. `dendritic/experiments/confidence/experiment.py` (to pass tokenizer)
5. Possibly: `dendritic/experiments/confidence/TrainingStrategy.py`

## Dependencies

- Requires tokenizer to be available (GPT2Tokenizer from transformers)
- May need to update imports in various files
- Ensure torch version compatibility