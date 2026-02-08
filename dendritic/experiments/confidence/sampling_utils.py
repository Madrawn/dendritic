"""
Sampling utilities for MiniGPT and ConfidenceAwareGPT models.

This module provides autoregressive sampling for models that don't have
built-in `.generate()` methods.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def sample_tokens_from_model(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cuda",
    use_confidence: bool = False,
) -> tuple[str, list[float] | None]:
    """
    Sample tokens from a MiniGPT or ConfidenceAwareGPT model using autoregressive generation.

    For ConfidenceAwareGPT models with use_confidence=True:
    - Start with confidence_scalars=None (model uses zeros)
    - For each generated token, get confidence prediction from model
    - Use that confidence prediction as input for next token (shifted right by 1)
    - Repeat

    Args:
        model: MiniGPT or ConfidenceAwareGPT model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Text prompt to start generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Top-p (nucleus) sampling parameter
        device: Device to run generation on
        use_confidence: Whether to use confidence predictions for ConfidenceAwareGPT models.
                       If True, uses confidence predictions from previous step as input.
                       If False, uses default zeros (None).

    Returns:
        tuple: (generated_text, confidence_predictions)
        - generated_text: Generated text (including prompt)
        - confidence_predictions: List of confidence predictions for each generated token,
                                 or None if model is not ConfidenceAwareGPT
    """
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    # Track confidence predictions if model is ConfidenceAwareGPT
    is_confidence_model = hasattr(model, "confidence_predictor")
    confidence_predictions_list = [] if is_confidence_model else None

    # Initialize confidence scalars for the first step
    # For confidence models with use_confidence=False, we pass None (model uses zeros)
    # For confidence models with use_confidence=True, we'll build confidence scalars as we generate
    confidence_scalars = None
    model_max_len = model.max_seq_len
    current_len = input_ids.shape[1]
    if current_len + max_new_tokens > model_max_len:
        max_new_tokens = model_max_len - current_len
        logging.warning(f"Truncating generation to {max_new_tokens} tokens to respect max_seq_len={model_max_len}")
    if max_new_tokens <= 0:
        return tokenizer.decode(input_ids[0]), None
    if max_new_tokens > 1000:  # Reasonable upper limit
        logging.warning(f"max_new_tokens ({max_new_tokens}) is very large, may cause memory issues")
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Get model output
            if hasattr(model, "forward") and callable(model.forward):
                if is_confidence_model:
                    # ConfidenceAwareGPT returns dict with logits and confidence_pred
                    outputs = model(generated, confidence_scalars=confidence_scalars)
                    logits = outputs["logits"]
                    confidence_pred = outputs["confidence_pred"]

                    # Store confidence prediction for the last position
                    if confidence_pred is not None and confidence_predictions_list is not None:
                        last_conf = confidence_pred[:, -1].item()
                        confidence_predictions_list.append(last_conf)

                        # For next step, prepare confidence scalars
                        if use_confidence:
                            # Create confidence scalars tensor for NEXT forward call
                            # The next forward call will have seq_len + 1 tokens (current + new token)
                            next_seq_len = generated.shape[1] + 1
                            conf_tensor = torch.zeros((1, next_seq_len, 1), device=device, dtype=torch.float32)

                            # Fill with previous confidence predictions (shifted right by 1)
                            # Position j gets confidence prediction from position j-1
                            # We have confidence predictions for positions 0..(seq_len-1)
                            # These need to go to positions 1..seq_len in the next forward call
                            if len(confidence_predictions_list) > 0:
                                fill_length = min(len(confidence_predictions_list), next_seq_len - 1)
                                conf_tensor[0, 1 : 1 + fill_length, 0] = torch.tensor(
                                    confidence_predictions_list[:fill_length], device=device, dtype=torch.float32
                                )
                            confidence_scalars = conf_tensor
                else:
                    # MiniGPT returns logits directly
                    logits = model(generated)
            else:
                raise ValueError(f"Model doesn't have a forward method: {type(model)}")

            # Get logits for the last token
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

            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if we generate EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the full sequence
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return full_text, confidence_predictions_list


def sample_model_output(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.95,
    use_confidence: bool = False,
) -> tuple[str, list[float] | None]:
    """
    Generate sample output from model and return as string.

    This is a convenience wrapper around sample_tokens_from_model that
    handles logging and error handling.

    Args:
        model: Model to sample from
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        use_confidence: Whether to use confidence predictions for ConfidenceAwareGPT models

    Returns:
        tuple: (generated_text, confidence_predictions)
        - generated_text: Generated text
        - confidence_predictions: List of confidence predictions or None
    """
    try:
        generated, confidence_predictions = sample_tokens_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            use_confidence=use_confidence,
        )
        return generated, confidence_predictions
    except Exception as e:
        logger.error(f"Error during sampling: {e}")
        return f"[Sampling error: {e}]", None
