"""
Sampling utilities for MiniGPT and DoubtAwareGPT models.

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
    MiniGPT or DoubtAwareGPT models.
    """

    device: str = "cuda"
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 0.95
    use_doubt: bool = False
    include_doubt_formatting: bool = True


def sample_tokens_from_model(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    config: SamplingConfig,
) -> tuple[str, list[int], list[int]]:
    """
    Sample tokens from a MiniGPT or DoubtAwareGPT model using autoregressive generation.

    For DoubtAwareGPT models with config.use_doubt=True:
    - Start with doubt_scalars=None (model uses zeros)
    - For each generated token, get doubt prediction from model
    - Use that doubt prediction as input for next token (shifted right by 1)
    - Repeat

    Args:
        model: MiniGPT or DoubtAwareGPT model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Text prompt to start generation
        config: Sampling configuration (SamplingConfig)

    Returns:
        tuple: (generated_text, generated_token_ids, full_token_ids)
        - generated_text: Generated text (including prompt)
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

    # Initialize loss prediction scalars for the first step
    # For doubt models with config.use_doubt=False, we pass None (model uses zeros)
    # For doubt models with config.use_doubt=True, we'll build loss prediction scalars as we generate
    model_max_len = model.max_seq_len
    current_len = input_ids.shape[1]
    effective_max_new_tokens = config.max_new_tokens
    if current_len + effective_max_new_tokens > model_max_len:
        effective_max_new_tokens = model_max_len - current_len
        logging.warning(
            f"Truncating generation to {effective_max_new_tokens} tokens to respect max_seq_len={model_max_len}"
        )
    if effective_max_new_tokens <= 0:
        return tokenizer.decode(input_ids[0]), [], input_ids[0].tolist()
    if effective_max_new_tokens > 1000:  # Reasonable upper limit
        logging.warning(f"max_new_tokens ({effective_max_new_tokens}) is very large, may cause memory issues")
    with torch.no_grad():
        for _ in range(effective_max_new_tokens):
            # Get model output
            logits, loss_pred_tensor = _get_model_logits(model, generated)

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
    return full_text, generated_token_ids, full_token_ids


def _get_model_logits(model: nn.Module, generated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Get logits from model, handling both doubt and non-doubt models."""
    if not hasattr(model, "forward") or not callable(model.forward):
        raise ValueError(f"Model doesn't have a forward method: {type(model)}")

    else:
        logits = model(generated)
        return logits, None


def _update_doubt_scalars_after_append(
    loss_prediction_vectors: list[torch.Tensor], next_seq_len: int, device: str, doubt_vector_dim: int
) -> torch.Tensor:
    """Update doubt scalars for the next forward pass after appending a token.

    Args:
        loss_prediction_vectors: List of vectors (each of shape [V]) from previous steps
        next_seq_len: The new sequence length after token append
        device: Device to create tensor on
        doubt_vector_dim: The dimension V of the doubt vector

    Returns:
        Tensor of shape [1, next_seq_len, V] with zeros at position 0 and vectors at positions 1..k
    """
    doubt_tensor = torch.zeros((1, next_seq_len, doubt_vector_dim), device=device, dtype=torch.float32)

    if len(loss_prediction_vectors) > 0:
        fill_length = min(len(loss_prediction_vectors), next_seq_len - 1)
        # Stack vectors into a tensor of shape [fill_length, V]
        vectors = torch.stack(loss_prediction_vectors[:fill_length], dim=0)
        # Assign to positions 1..fill_length+1
        doubt_tensor[0, 1 : 1 + fill_length, :] = vectors
    return doubt_tensor


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


def format_tokens_with_doubt(
    tokenizer, generated_token_ids: list[int], loss_predictions: list[float], doubt_precision: int = 1
) -> str:
    """
    Format tokens with loss predictions in parentheses.

    Args:
        tokenizer: Tokenizer with decode method
        generated_token_ids: List of token IDs for generated tokens
        loss_predictions: List of loss prediction values for each generated token
        doubt_precision: Number of decimal places for loss predictions

    Returns:
        Formatted string like "token1(8.8) token2(7.1)"
    """
    if len(generated_token_ids) != len(loss_predictions):
        raise ValueError(
            f"Token count ({len(generated_token_ids)}) doesn't match loss prediction count ({len(loss_predictions)})"
        )

    formatted_parts = []
    for token_id, loss_pred in zip(generated_token_ids, loss_predictions):
        # Decode single token
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        # Clean up whitespace (tokenizer.decode might add spaces)
        token_text = token_text.strip()
        # Format with loss prediction
        loss_str = f"{loss_pred:.{doubt_precision}f}"
        formatted_parts.append(f"{token_text}({loss_str})")

    return " ".join(formatted_parts)


def sample_model_output(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    config: SamplingConfig,
) -> str:
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
        str: Generated text
    """
    try:
        generated, generated_token_ids, full_token_ids = sample_tokens_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
        )

        return generated

    except Exception as e:
        logger.error(f"Error during sampling: {e}")
        raise e
