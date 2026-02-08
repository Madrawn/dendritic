# Confidence Token Pairing Improvement - Implementation Todo

## Overview
Improve the logged sampling output to show each generated token with its confidence score in brackets, like `Once upon a time was(8.8) a(7.1)` etc. This makes it easier to interpret which tokens the model is confident about.

## Requirements Clarified
1. **Attach confidence to each generated token** (subword tokens), skip prompt tokens
2. **Use 1 decimal place** for confidence scores
3. **Add as new line** 'Sampled Tokens with Confidence:' in the output file
4. **Keep old format** for backward compatibility and machine readability

## Current State Analysis

### Current Flow:
1. `UnifiedTrainer._sample_and_log()` calls `sample_model_output()`
2. `sample_model_output()` calls `sample_tokens_from_model()`
3. `sample_tokens_from_model()` returns:
   - `full_text`: Complete generated text (prompt + generated tokens) as single string
   - `confidence_predictions_list`: List of confidence values for each **generated** token only

### Challenge:
We need to align generated tokens with their confidence predictions. The prompt tokens don't have confidence predictions, and we need to decode tokens individually to pair them with confidence scores.

## Implementation Plan

### Phase 1: Modify `sampling_utils.py` to Return Token-Level Information

**File**: `dendritic/experiments/confidence/sampling_utils.py`

**Changes**:
1. Modify `sample_tokens_from_model()` to return additional information:
   - `generated_token_ids`: List of token IDs for generated tokens only
   - `full_token_ids`: Complete sequence of token IDs (prompt + generated)
   - Keep existing `full_text` and `confidence_predictions_list` returns

2. Create helper function `format_tokens_with_confidence()` that:
   - Takes tokenizer, generated_token_ids, and confidence_predictions_list
   - Decodes each token individually
   - Formats as `token(confidence)` pairs
   - Handles special tokens and whitespace properly

3. Update `sample_model_output()` to use the new formatting

### Phase 2: Update `UnifiedTrainer.py` to Use New Format

**File**: `dendritic/experiments/confidence/UnifiedTrainer.py`

**Changes**:
1. Modify `_save_sampled_tokens_to_file()` to:
   - Accept new `formatted_tokens_with_confidence` parameter
   - Write new line "Sampled Tokens with Confidence: {formatted_tokens_with_confidence}"
   - Keep existing "Sampled Tokens:" line for backward compatibility

2. Update `_sample_and_log()` to:
   - Get formatted tokens with confidence from sampling utility
   - Pass them to `_save_sampled_tokens_to_file()`

### Phase 3: Update Tests

**File**: `tests/test_confidence_experiment/test_sampling.py`

**Changes**:
1. Update mock tokenizer to support `decode()` for single tokens
2. Add tests for new token-confidence formatting
3. Ensure backward compatibility with existing tests

## Detailed Implementation Steps

### Step 1: Modify `sampling_utils.py`

```python
def sample_tokens_from_model(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cuda",
    use_confidence: bool = False,
) -> tuple[str, list[float] | None, list[int] | None, list[int] | None]:
    """
    Sample tokens from a MiniGPT or ConfidenceAwareGPT model using autoregressive generation.
    
    Returns:
        tuple: (full_text, confidence_predictions, generated_token_ids, full_token_ids)
        - full_text: Generated text (including prompt)
        - confidence_predictions: List of confidence predictions for each generated token,
                                 or None if model is not ConfidenceAwareGPT
        - generated_token_ids: List of token IDs for generated tokens only
        - full_token_ids: Complete sequence of token IDs (prompt + generated)
    """
    # Existing code with modifications:
    # 1. Store generated_token_ids list
    # 2. Store full_token_ids (input_ids + generated_token_ids)
    # 3. Return additional values
```

```python
def format_tokens_with_confidence(
    tokenizer,
    generated_token_ids: list[int],
    confidence_predictions: list[float],
    confidence_precision: int = 1
) -> str:
    """
    Format tokens with confidence scores in parentheses.
    
    Args:
        tokenizer: Tokenizer with decode method
        generated_token_ids: List of token IDs for generated tokens
        confidence_predictions: List of confidence scores for each generated token
        confidence_precision: Number of decimal places for confidence scores
    
    Returns:
        Formatted string like "token1(8.8) token2(7.1)"
    """
    if len(generated_token_ids) != len(confidence_predictions):
        raise ValueError(f"Token count ({len(generated_token_ids)}) doesn't match confidence count ({len(confidence_predictions)})")
    
    formatted_parts = []
    for token_id, confidence in zip(generated_token_ids, confidence_predictions):
        # Decode single token
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        # Clean up whitespace (tokenizer.decode might add spaces)
        token_text = token_text.strip()
        # Format with confidence
        confidence_str = f"{confidence:.{confidence_precision}f}"
        formatted_parts.append(f"{token_text}({confidence_str})")
    
    return " ".join(formatted_parts)
```

### Step 2: Update `sample_model_output()` to Use Formatting

```python
def sample_model_output(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.95,
    use_confidence: bool = False,
    include_confidence_formatting: bool = True,
) -> tuple[str, list[float] | None, str | None]:
    """
    Generate sample output from model with optional confidence formatting.
    
    Returns:
        tuple: (generated_text, confidence_predictions, formatted_tokens_with_confidence)
        - generated_text: Generated text
        - confidence_predictions: List of confidence predictions or None
        - formatted_tokens_with_confidence: Formatted tokens with confidence scores,
                                           or None if no confidence predictions
    """
    try:
        generated, confidence_predictions, generated_token_ids, full_token_ids = sample_tokens_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            use_confidence=use_confidence,
        )
        
        formatted_tokens = None
        if include_confidence_formatting and confidence_predictions is not None:
            formatted_tokens = format_tokens_with_confidence(
                tokenizer=tokenizer,
                generated_token_ids=generated_token_ids,
                confidence_predictions=confidence_predictions,
                confidence_precision=1
            )
        
        return generated, confidence_predictions, formatted_tokens
        
    except Exception as e:
        logger.error(f"Error during sampling: {e}")
        return f"[Sampling error: {e}]", None, None
```

### Step 3: Update `UnifiedTrainer._sample_and_log()`

```python
def _sample_and_log(self, model: nn.Module, step: int, eval_loss: float, seed: int):
    """
    Sample tokens from model and log results.
    """
    if self.tokenizer is None:
        logging.warning("No tokenizer provided, skipping sampling")
        return

    try:
        from dendritic.experiments.confidence.sampling_utils import (
            sample_model_output,
        )

        # Determine if we should use confidence-aware sampling
        use_confidence = hasattr(model, "confidence_predictor")

        generated, confidence_predictions, formatted_tokens_with_confidence = sample_model_output(
            model=model,
            tokenizer=self.tokenizer,
            prompt=self.config.sampling_prompt,
            device=self.device,
            max_new_tokens=self.config.sampling_max_tokens,
            temperature=self.config.sampling_temperature,
            top_p=self.config.sampling_top_p,
            use_confidence=use_confidence,
            include_confidence_formatting=True,
        )

        # Log truncated version to console
        truncated = generated[:100] + "..." if len(generated) > 100 else generated
        log_message = f"{self.model_type} seed={seed} step={step}: eval_loss={eval_loss:.4f}, sampled: {truncated}"

        # Add confidence predictions to log if available
        if confidence_predictions is not None and len(confidence_predictions) > 0:
            avg_conf = sum(confidence_predictions) / len(confidence_predictions)
            log_message += f", avg_conf={avg_conf:.4f}"

        logging.info(log_message)

        # Save full version to text file
        self._save_sampled_tokens_to_file(
            step=step,
            seed=seed,
            model_type=self.model_type,
            eval_loss=eval_loss,
            sampled_text=generated,
            confidence_predictions=confidence_predictions,
            formatted_tokens_with_confidence=formatted_tokens_with_confidence,
        )

    except Exception as e:
        logging.warning(f"Sampling failed: {e}")
```

### Step 4: Update `UnifiedTrainer._save_sampled_tokens_to_file()`

```python
def _save_sampled_tokens_to_file(
    self,
    step,
    seed,
    model_type,
    eval_loss,
    sampled_text,
    confidence_predictions=None,
    formatted_tokens_with_confidence=None,
):
    """
    Save sampled tokens to text file alongside JSON results.
    """
    from datetime import datetime
    from pathlib import Path

    # Ensure results directory exists
    results_path = Path(self.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Create text file name based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_file = results_path / f"{timestamp}_{model_type}_seed{seed}_sampled_tokens.txt"

    # Append to file
    with open(text_file, "a", encoding="utf-8") as f:
        f.write(f"=== Step {step} (Seed: {seed}, Model: {model_type}) ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Eval Loss: {eval_loss:.4f}\n")
        f.write(f"Sampled Tokens: {sampled_text}\n")

        # Add confidence predictions if available
        if confidence_predictions is not None and len(confidence_predictions) > 0:
            f.write(f"Confidence Predictions: {confidence_predictions}\n")
            f.write(f"Avg Confidence: {sum(confidence_predictions) / len(confidence_predictions):.4f}\n")
            
            # Add formatted tokens with confidence if available
            if formatted_tokens_with_confidence is not None:
                f.write(f"Sampled Tokens with Confidence: {formatted_tokens_with_confidence}\n")

        f.write("=" * 40 + "\n\n")
```

### Step 5: Update Tests

1. **Update MockTokenizer** in `test_sampling.py`:
   - Add proper `decode()` method that handles single tokens
   - Ensure it returns appropriate text for token IDs

2. **Add new test cases**:
   - Test `format_tokens_with_confidence()` function
   - Test updated `sample_tokens_from_model()` with token ID returns
   - Test integration with `UnifiedTrainer`

## Testing Checklist

- [ ] Run existing tests to ensure no regression
- [ ] Test confidence model sampling produces formatted output
- [ ] Test standard model sampling (no confidence) works unchanged
- [ ] Verify output file contains new "Sampled Tokens with Confidence:" line
- [ ] Check formatting: tokens(confidence) with 1 decimal place
- [ ] Verify prompt tokens are excluded from confidence formatting
- [ ] Test edge cases: empty confidence list, single token, special tokens

## Files to Modify

1. `dendritic/experiments/confidence/sampling_utils.py`
2. `dendritic/experiments/confidence/UnifiedTrainer.py`
3. `tests/test_confidence_experiment/test_sampling.py`

## Dependencies

- Requires tokenizer with proper `decode()` method for single tokens
- Maintains backward compatibility with existing code
- No new external dependencies

## Potential Issues and Solutions

1. **Tokenization differences**: Some tokenizers add spaces when decoding single tokens. Solution: strip whitespace in `format_tokens_with_confidence()`.

2. **Special tokens**: EOS token or other special tokens might not decode to readable text. Solution: Use `skip_special_tokens=True` in decode.

3. **Performance impact**: Decoding each token individually adds overhead. Solution: This is only done during evaluation sampling, not training.

4. **Memory**: Storing token IDs adds minimal memory overhead.

## Example Output

After implementation, the output file should contain:

```
=== Step 440 (Seed: 42, Model: confidence) ===
Timestamp: 2026-02-08T13:55:49.594193
Eval Loss: 5.4751
Sampled Tokens: Once upon a time was a old how again. the day. Tom smiled. there there's be so he said, was the day, there. Jack. Lily. " wild. his the and enjoyed it at the, the different there in the the day, Jane
Confidence Predictions: [8.597891807556152, 7.165717124938965, ...]
Avg Confidence: 6.1550
Sampled Tokens with Confidence: Once(8.6) upon(7.2) a(8.4) time(6.8) was(5.9) a(5.4) old(6.3) how(7.4) again(4.8) ...
========================================