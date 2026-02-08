"""
Tests for token sampling functionality in confidence experiments.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from dendritic.experiments.confidence.sampling_utils import (
    sample_tokens_from_model,
    sample_model_output,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 50256

    def encode(self, text, return_tensors="pt"):
        """Mock encode method."""
        return torch.tensor([[1, 2, 3, 4]])

    def decode(self, tokens, skip_special_tokens=True):
        """Mock decode method."""
        return "Mock generated text"


class MockMiniGPT(nn.Module):
    """Mock MiniGPT model for testing."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 1000

    def forward(self, input_ids):
        """Mock forward method."""
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, self.vocab_size)


class MockConfidenceAwareGPT(MockMiniGPT):
    """Mock ConfidenceAwareGPT model for testing."""

    def __init__(self):
        super().__init__()
        self.confidence_predictor = nn.Linear(10, 1)

    def forward(self, input_ids, confidence_scalars=None):
        """Mock forward method."""
        batch_size, seq_len = input_ids.shape
        return {
            "logits": torch.randn(batch_size, seq_len, self.vocab_size),
            "confidence_pred": torch.randn(batch_size, seq_len),
        }


@pytest.mark.unit
def test_sample_tokens_from_model_minigpt():
    """Test sampling from MiniGPT model."""
    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    # Test basic sampling
    text, confidence_predictions = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        max_new_tokens=5,
        temperature=1.0,
        top_p=0.95,
        device="cpu",
    )

    assert isinstance(text, str)
    assert "Mock generated text" in text
    assert confidence_predictions is None  # MiniGPT doesn't have confidence predictions


@pytest.mark.unit
def test_sample_tokens_from_model_confidence_aware():
    """Test sampling from ConfidenceAwareGPT model."""
    model = MockConfidenceAwareGPT()
    tokenizer = MockTokenizer()

    # Test basic sampling
    text, confidence_predictions = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        max_new_tokens=5,
        temperature=1.0,
        top_p=0.95,
        device="cpu",
    )

    assert isinstance(text, str)
    assert "Mock generated text" in text
    assert confidence_predictions is not None
    assert (
        len(confidence_predictions) == 5
    )  # Should have 5 confidence predictions for 5 new tokens


@pytest.mark.unit
def test_sample_tokens_from_model_top_p_sampling():
    """Test top-p (nucleus) sampling."""
    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    # Test with top_p < 1.0
    text, confidence_predictions = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        max_new_tokens=5,
        temperature=0.8,
        top_p=0.9,
        device="cpu",
    )

    assert isinstance(text, str)
    assert confidence_predictions is None


@pytest.mark.unit
def test_sample_tokens_from_model_temperature():
    """Test temperature scaling."""
    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    # Test with different temperatures
    text_low_temp, conf_low = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        max_new_tokens=5,
        temperature=0.5,
        top_p=0.95,
        device="cpu",
    )

    text_high_temp, conf_high = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        max_new_tokens=5,
        temperature=1.5,
        top_p=0.95,
        device="cpu",
    )

    assert isinstance(text_low_temp, str)
    assert isinstance(text_high_temp, str)
    assert conf_low is None
    assert conf_high is None


@pytest.mark.unit
def test_sample_model_output_error_handling():
    """Test error handling in sample_model_output."""
    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    # Mock an error during sampling
    with patch.object(model, "forward", side_effect=RuntimeError("Test error")):
        text, confidence_predictions = sample_model_output(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            device="cpu",
        )

    assert "[Sampling error:" in text
    assert confidence_predictions is None


@pytest.mark.unit
def test_sample_tokens_from_model_eos_stopping():
    """Test that sampling stops at EOS token."""
    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    # Mock tokenizer to return EOS token
    original_decode = tokenizer.decode
    call_count = 0

    def mock_decode(tokens, skip_special_tokens=True):
        nonlocal call_count
        call_count += 1
        if call_count > 3:  # Simulate EOS after 3 tokens
            return ""
        return "Token"

    tokenizer.decode = mock_decode

    try:
        text, confidence_predictions = sample_tokens_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            max_new_tokens=10,
            temperature=1.0,
            top_p=0.95,
            device="cpu",
        )
        assert isinstance(text, str)
        assert confidence_predictions is None
    finally:
        tokenizer.decode = original_decode


@pytest.mark.unit
def test_confidence_aware_gpt_forward_calls():
    """Test that ConfidenceAwareGPT forward is called with correct confidence scalars."""
    # Create a mock model that tracks forward calls
    forward_calls = []

    class TrackingConfidenceAwareGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 1000
            self.confidence_predictor = nn.Linear(10, 1)

        def forward(self, input_ids, confidence_scalars=None):
            # Track the call
            forward_calls.append({
                "input_ids_shape": input_ids.shape,
                "confidence_scalars_shape": confidence_scalars.shape
                if confidence_scalars is not None
                else None,
                "confidence_scalars_values": confidence_scalars.tolist()
                if confidence_scalars is not None
                else None,
            })

            batch_size, seq_len = input_ids.shape
            # Return increasing confidence predictions to verify they're used
            return {
                "logits": torch.randn(batch_size, seq_len, self.vocab_size),
                "confidence_pred": torch.arange(seq_len, dtype=torch.float32).view(
                    1, seq_len
                )
                * 0.1
                + 0.5,
            }

    model = TrackingConfidenceAwareGPT()
    tokenizer = MockTokenizer()

    # Test with use_confidence=True
    text, confidence_predictions = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        max_new_tokens=3,  # Generate 3 new tokens
        temperature=1.0,
        top_p=0.95,
        device="cpu",
        use_confidence=True,
    )

    # Verify forward calls were made
    assert len(forward_calls) == 3  # Should be called 3 times (once for each new token)

    # Check first call: initial prompt (4 tokens) with no confidence scalars
    first_call = forward_calls[0]
    assert first_call["input_ids_shape"] == (1, 4)  # Prompt tokens
    assert first_call["confidence_scalars_shape"] is None  # First call should have None

    # Check second call: prompt + 1 generated token (5 tokens total)
    second_call = forward_calls[1]
    assert second_call["input_ids_shape"] == (1, 5)  # Prompt + 1 generated token

    # Second call should have confidence scalars with shape [1, 5, 1]
    # First position should be 0, subsequent positions should have confidence predictions
    assert second_call["confidence_scalars_shape"] == (1, 5, 1)

    # Check third call: prompt + 2 generated tokens (6 tokens total)
    third_call = forward_calls[2]
    assert third_call["input_ids_shape"] == (1, 6)
    assert third_call["confidence_scalars_shape"] == (1, 6, 1)

    # Verify confidence predictions were collected
    assert confidence_predictions is not None
    assert len(confidence_predictions) == 3  # 3 new tokens

    # Verify the confidence predictions match what we'd expect from our mock
    # Our mock returns confidence_pred = [0.5, 0.6, 0.7, 0.8] for 4 tokens
    # The last position (position 3) would be 0.8 for the first generated token
    # Then for next token, position 4 would be based on previous predictions
    # This is a complex check, but we can at least verify they're numbers
    assert all(isinstance(c, float) for c in confidence_predictions)


@pytest.mark.unit
def test_confidence_aware_gpt_confidence_values():
    """Test that confidence values are properly propagated through sampling."""
    # Create a mock model that returns predictable confidence values
    forward_calls = []

    class PredictableConfidenceAwareGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 1000
            self.confidence_predictor = nn.Linear(10, 1)

        def forward(self, input_ids, confidence_scalars=None):
            # Track the call with more detail
            call_info = {
                "input_ids_shape": input_ids.shape,
                "confidence_scalars": confidence_scalars,
            }

            if confidence_scalars is not None:
                call_info["confidence_scalars_shape"] = confidence_scalars.shape
                call_info["confidence_scalars_values"] = confidence_scalars.tolist()
                # Verify that confidence scalars are properly shifted
                # Position 0 should always be 0
                assert confidence_scalars[0, 0, 0].item() == 0.0, (
                    f"Position 0 should be 0, got {confidence_scalars[0, 0, 0].item()}"
                )

            forward_calls.append(call_info)

            batch_size, seq_len = input_ids.shape
            # Return predictable confidence predictions: 0.1 * position + 0.5
            confidence_pred = (
                torch.arange(seq_len, dtype=torch.float32).view(1, seq_len) * 0.1 + 0.5
            )

            return {
                "logits": torch.randn(batch_size, seq_len, self.vocab_size),
                "confidence_pred": confidence_pred,
            }

    model = PredictableConfidenceAwareGPT()
    tokenizer = MockTokenizer()

    # Test with use_confidence=True
    text, confidence_predictions = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        max_new_tokens=3,  # Generate 3 new tokens
        temperature=1.0,
        top_p=0.95,
        device="cpu",
        use_confidence=True,
    )

    # Verify we made the expected number of calls
    assert len(forward_calls) == 3

    # Check that confidence predictions were collected
    assert confidence_predictions is not None
    assert len(confidence_predictions) == 3

    # The mock returns confidence_pred = [0.5, 0.6, 0.7, 0.8] for 4 tokens
    # First generated token gets confidence from position 3: 0.8
    # Second generated token should get confidence from previous prediction
    # Let's trace through the logic:
    # 1. First call: 4 tokens, confidence_scalars=None
    #    Returns confidence_pred = [0.5, 0.6, 0.7, 0.8]
    #    Stores last_conf = 0.8
    #    Creates confidence_scalars for next call: [0, 0.5, 0.6, 0.7, 0] (5 positions)
    # 2. Second call: 5 tokens, confidence_scalars = [0, 0.5, 0.6, 0.7, 0]
    #    Returns confidence_pred = [0.5, 0.6, 0.7, 0.8, 0.9] (5 positions)
    #    Stores last_conf = 0.9
    #    Creates confidence_scalars for next call: [0, 0.5, 0.6, 0.7, 0.8, 0] (6 positions)
    # 3. Third call: 6 tokens, confidence_scalars = [0, 0.5, 0.6, 0.7, 0.8, 0]
    #    Returns confidence_pred = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] (6 positions)
    #    Stores last_conf = 1.0

    # So confidence_predictions should be [0.8, 0.9, 1.0]
    # Allow some tolerance for floating point
    expected_confidences = [0.8, 0.9, 1.0]
    for i, (actual, expected) in enumerate(
        zip(confidence_predictions, expected_confidences)
    ):
        assert abs(actual - expected) < 0.01, (
            f"Confidence prediction {i}: expected {expected}, got {actual}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
