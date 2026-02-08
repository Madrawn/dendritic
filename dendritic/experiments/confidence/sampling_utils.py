"""
Sampling utilities for MiniGPT and ConfidenceAwareGPT models.

This module provides autoregressive sampling for models that don't have
built-in `.generate()` methods.
"""

import torch
import torch.nn as nn
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for token sampling.

    Encapsulates all parameters needed for autoregressive sampling from
    MiniGPT or ConfidenceAwareGPT models.
    """

    device: str = "cuda"
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 0.95
    use_confidence: bool = False
    include_confidence_formatting: bool = True


def sample_tokens_from_model(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    config: SamplingConfig,
) -> tuple[str, list[float] | None, list[int], list[int]]:
    """
    Sample tokens from a MiniGPT or ConfidenceAwareGPT model using autoregressive generation.

    For ConfidenceAwareGPT models with config.use_confidence=True:
    - Start with confidence_scalars=None (model uses zeros)
    - For each generated token, get confidence prediction from model
    - Use that confidence prediction as input for next token (shifted right by 1)
    - Repeat

    Args:
        model: MiniGPT or ConfidenceAwareGPT model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Text prompt to start generation
        config: Sampling configuration (SamplingConfig)

    Returns:
        tuple: (generated_text, confidence_predictions, generated_token_ids, full_token_ids)
        - generated_text: Generated text (including prompt)
        - confidence_predictions: List of confidence predictions for each generated token,
                                 or None if model is not ConfidenceAwareGPT
        - generated_token_ids: List of token IDs for generated tokens only
        - full_token_ids: Complete sequence of token IDs (prompt + generated)
    """
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    generated = input_ids.clone()

    # Track generated token IDs (excluding prompt tokens)
    generated_token_ids = []
    full_token_ids = input_ids[0].tolist()  # Start with prompt token IDs

    # Track confidence predictions if model is ConfidenceAwareGPT
    is_confidence_model = hasattr(model, "confidence_predictor")
    confidence_predictions_list = [] if is_confidence_model else None

    # Initialize confidence scalars for the first step
    # For confidence models with config.use_confidence=False, we pass None (model uses zeros)
    # For confidence models with config.use_confidence=True, we'll build confidence scalars as we generate
    confidence_scalars = None
    model_max_len = model.max_seq_len
    current_len = input_ids.shape[1]
    effective_max_new_tokens = config.max_new_tokens
    if current_len + effective_max_new_tokens > model_max_len:
        effective_max_new_tokens = model_max_len - current_len
        logging.warning(
            f"Truncating generation to {effective_max_new_tokens} tokens to respect max_seq_len={model_max_len}"
        )
    if effective_max_new_tokens <= 0:
        return tokenizer.decode(input_ids[0]), None, [], input_ids[0].tolist()
    if effective_max_new_tokens > 1000:  # Reasonable upper limit
        logging.warning(f"max_new_tokens ({effective_max_new_tokens}) is very large, may cause memory issues")
    with torch.no_grad():
        for _ in range(effective_max_new_tokens):
            # Get model output
            logits, confidence_value = _get_model_logits(model, generated, is_confidence_model, confidence_scalars)

            # Store confidence prediction if available
            if confidence_value is not None and confidence_predictions_list is not None:
                confidence_predictions_list.append(confidence_value)

            # Update confidence scalars for next iteration if using confidence model
            if is_confidence_model and config.use_confidence:
                # confidence_predictions_list is guaranteed to be not None when is_confidence_model is True
                assert confidence_predictions_list is not None
                confidence_scalars = _update_confidence_scalars_after_append(
                    confidence_predictions_list, generated.shape[1] + 1, config.device
                )

            # Sample next token
            next_token = _sample_next_token(logits, config.temperature, config.top_p)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Track generated token IDs
            generated_token_ids.append(next_token.item())
            full_token_ids.append(next_token.item())

            # Stop if we generate EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the full sequence
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return full_text, confidence_predictions_list, generated_token_ids, full_token_ids


def _get_model_logits(
    model: nn.Module, generated: torch.Tensor, is_confidence_model: bool, confidence_scalars: torch.Tensor | None
) -> tuple[torch.Tensor, float | None]:
    """Get logits from model, handling both confidence and non-confidence models."""
    if not hasattr(model, "forward") or not callable(model.forward):
        raise ValueError(f"Model doesn't have a forward method: {type(model)}")

    if is_confidence_model:
        outputs = model(generated, confidence_scalars=confidence_scalars)
        logits = outputs["logits"]
        confidence_pred = outputs["confidence_pred"]
        last_conf = None
        if confidence_pred is not None:
            last_conf = confidence_pred[:, -1].item()
        return logits, last_conf
    else:
        logits = model(generated)
        return logits, None


def _update_confidence_scalars_after_append(
    confidence_predictions_list: list, next_seq_len: int, device: str
) -> torch.Tensor:
    """Update confidence scalars for the next forward pass after appending a token."""
    conf_tensor = torch.zeros((1, next_seq_len, 1), device=device, dtype=torch.float32)

    if len(confidence_predictions_list) > 0:
        fill_length = min(len(confidence_predictions_list), next_seq_len - 1)
        conf_tensor[0, 1 : 1 + fill_length, 0] = torch.tensor(
            confidence_predictions_list[:fill_length], device=device, dtype=torch.float32
        )
    return conf_tensor


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample next token from logits using temperature and top-p filtering."""
    next_token_logits = logits[:, -1, :] / temperature

    # Apply top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create mask for indices to remove
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[..., indices_to_remove] = -float("inf")

    # Sample from the distribution
    probs = torch.softmax(next_token_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def format_tokens_with_confidence(
    tokenizer, generated_token_ids: list[int], confidence_predictions: list[float], confidence_precision: int = 1
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
        raise ValueError(
            f"Token count ({len(generated_token_ids)}) doesn't match confidence count ({len(confidence_predictions)})"
        )

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


def sample_model_output(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    config: SamplingConfig,
) -> tuple[str, list[float] | None, str | None]:
    """
    Generate sample output from model and return as string.

    This is a convenience wrapper around sample_tokens_from_model that
    handles logging and error handling.

    Args:
        model: Model to sample from
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt
        config: Sampling configuration (SamplingConfig)

    Returns:
        tuple: (generated_text, confidence_predictions, formatted_tokens_with_confidence)
        - generated_text: Generated text
        - confidence_predictions: List of confidence predictions or None
        - formatted_tokens_with_confidence: Formatted tokens with confidence scores or None
    """
    try:
        generated, confidence_predictions, generated_token_ids, full_token_ids = sample_tokens_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
        )

        formatted_tokens = None
        if config.include_confidence_formatting and confidence_predictions is not None:
            formatted_tokens = format_tokens_with_confidence(
                tokenizer=tokenizer,
                generated_token_ids=generated_token_ids,
                confidence_predictions=confidence_predictions,
                confidence_precision=1,
            )

        return generated, confidence_predictions, formatted_tokens

    except Exception as e:
        logger.error(f"Error during sampling: {e}")
        return f"[Sampling error: {e}]", None, None
