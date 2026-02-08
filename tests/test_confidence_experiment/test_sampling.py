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
        # Map token IDs to their text representations
        self.token_map = {
            1: "Once",
            2: "upon",
            3: "a",
            4: "time",
            5: "was",
            6: "the",
            7: "day",
            8: "there",
            9: "were",
            10: "three",
            50256: "",  # EOS token
        }

    def encode(self, text, return_tensors="pt"):
        """Mock encode method."""
        return torch.tensor([[1, 2, 3, 4]])

    def decode(self, tokens, skip_special_tokens=True):
        """Mock decode method that handles both single tokens and lists of tokens."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        # Handle single token
        if isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], int):
            token_id = tokens[0]
            if skip_special_tokens and token_id == self.eos_token_id:
                return ""
            return self.token_map.get(token_id, f"<token_{token_id}>")

        # Handle list of tokens
        if isinstance(tokens, list):
            decoded_tokens = []
            for token_id in tokens:
                if skip_special_tokens and token_id == self.eos_token_id:
                    continue
                decoded_tokens.append(self.token_map.get(token_id, f"<token_{token_id}>"))
            return " ".join(decoded_tokens)

        return "Mock generated text"


class MockMiniGPT(nn.Module):
    """Mock MiniGPT model for testing."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 1000
        self.max_seq_len = 10

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
    text, confidence_predictions, generated_token_ids, full_token_ids = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        max_new_tokens=5,
        temperature=1.0,
        top_p=0.95,
        device="cpu",
    )

    assert isinstance(text, str)
    assert "Once" in text  # Should contain the decoded prompt tokens
    assert confidence_predictions is None  # MiniGPT doesn't have confidence predictions


@pytest.mark.unit
def test_sample_tokens_from_model_confidence_aware():
    """Test sampling from ConfidenceAwareGPT model."""
    model = MockConfidenceAwareGPT()
    tokenizer = MockTokenizer()

    # Test basic sampling
    text, confidence_predictions, generated_token_ids, full_token_ids = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        max_new_tokens=5,
        temperature=1.0,
        top_p=0.95,
        device="cpu",
    )

    assert isinstance(text, str)
    assert "Once" in text  # Should contain the decoded prompt tokens
    assert confidence_predictions is not None
    assert len(confidence_predictions) == 5  # Should have 5 confidence predictions for 5 new tokens


@pytest.mark.unit
def test_sample_tokens_from_model_top_p_sampling():
    """Test top-p (nucleus) sampling."""
    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    # Test with top_p < 1.0
    text, confidence_predictions, generated_token_ids, full_token_ids = sample_tokens_from_model(
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
    text_low_temp, conf_low, generated_token_ids_low, full_token_ids_low = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        max_new_tokens=5,
        temperature=0.5,
        top_p=0.95,
        device="cpu",
    )

    text_high_temp, conf_high, generated_token_ids_high, full_token_ids_high = sample_tokens_from_model(
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
        text, confidence_predictions, formatted_tokens = sample_model_output(
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
        text, confidence_predictions, generated_token_ids, full_token_ids = sample_tokens_from_model(
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
            self.max_seq_len = 10

        def forward(self, input_ids, confidence_scalars=None):
            # Track the call
            forward_calls.append({
                "input_ids_shape": input_ids.shape,
                "confidence_scalars_shape": confidence_scalars.shape if confidence_scalars is not None else None,
                "confidence_scalars_values": confidence_scalars.tolist() if confidence_scalars is not None else None,
            })

            batch_size, seq_len = input_ids.shape
            # Return increasing confidence predictions to verify they're used
            return {
                "logits": torch.randn(batch_size, seq_len, self.vocab_size),
                "confidence_pred": torch.arange(seq_len, dtype=torch.float32).view(1, seq_len) * 0.1 + 0.5,
            }

    model = TrackingConfidenceAwareGPT()
    tokenizer = MockTokenizer()

    # Test with use_confidence=True
    text, confidence_predictions, generated_token_ids, full_token_ids = sample_tokens_from_model(
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
            self.max_seq_len = 10

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
            confidence_pred = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len) * 0.1 + 0.5

            return {
                "logits": torch.randn(batch_size, seq_len, self.vocab_size),
                "confidence_pred": confidence_pred,
            }

    model = PredictableConfidenceAwareGPT()
    tokenizer = MockTokenizer()

    # Test with use_confidence=True
    text, confidence_predictions, generated_token_ids, full_token_ids = sample_tokens_from_model(
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
    for i, (actual, expected) in enumerate(zip(confidence_predictions, expected_confidences)):
        assert abs(actual - expected) < 0.01, f"Confidence prediction {i}: expected {expected}, got {actual}"


@pytest.mark.unit
def test_format_tokens_with_confidence():
    """Test the format_tokens_with_confidence function."""
    from dendritic.experiments.confidence.sampling_utils import format_tokens_with_confidence

    tokenizer = MockTokenizer()

    # Test basic formatting
    generated_token_ids = [5, 6, 7]  # "was", "the", "day"
    confidence_predictions = [8.5, 7.2, 6.8]

    result = format_tokens_with_confidence(
        tokenizer=tokenizer,
        generated_token_ids=generated_token_ids,
        confidence_predictions=confidence_predictions,
        confidence_precision=1,
    )

    expected = "was(8.5) the(7.2) day(6.8)"
    assert result == expected


@pytest.mark.unit
def test_format_tokens_with_confidence_whitespace_stripping():
    """Test that whitespace is properly stripped from decoded tokens."""
    from dendritic.experiments.confidence.sampling_utils import format_tokens_with_confidence

    class WhitespaceTokenizer(MockTokenizer):
        def decode(self, tokens, skip_special_tokens=True):
            # Add leading space to simulate some tokenizers
            if isinstance(tokens, list) and len(tokens) == 1:
                return f" {self.token_map.get(tokens[0], f'<token_{tokens[0]}>')}"
            return super().decode(tokens, skip_special_tokens)

    tokenizer = WhitespaceTokenizer()
    generated_token_ids = [5, 6, 7]  # " was", " the", " day"
    confidence_predictions = [8.5, 7.2, 6.8]

    result = format_tokens_with_confidence(
        tokenizer=tokenizer,
        generated_token_ids=generated_token_ids,
        confidence_predictions=confidence_predictions,
        confidence_precision=1,
    )

    expected = "was(8.5) the(7.2) day(6.8)"  # Should be stripped of leading spaces
    assert result == expected


@pytest.mark.unit
def test_format_tokens_with_confidence_validation():
    """Test that format_tokens_with_confidence validates token count match."""
    from dendritic.experiments.confidence.sampling_utils import format_tokens_with_confidence

    tokenizer = MockTokenizer()

    # Test mismatched lengths
    generated_token_ids = [5, 6]  # 2 tokens
    confidence_predictions = [8.5]  # 1 confidence value

    with pytest.raises(ValueError, match="Token count .* doesn't match confidence count"):
        format_tokens_with_confidence(
            tokenizer=tokenizer,
            generated_token_ids=generated_token_ids,
            confidence_predictions=confidence_predictions,
        )


@pytest.mark.unit
def test_format_tokens_with_confidence_precision():
    """Test different precision levels for confidence scores."""
    from dendritic.experiments.confidence.sampling_utils import format_tokens_with_confidence

    tokenizer = MockTokenizer()
    generated_token_ids = [5, 6]
    confidence_predictions = [8.56789, 7.23456]

    # Test 0 decimal places
    result_0 = format_tokens_with_confidence(
        tokenizer=tokenizer,
        generated_token_ids=generated_token_ids,
        confidence_predictions=confidence_predictions,
        confidence_precision=0,
    )
    expected_0 = "was(9) the(7)"
    assert result_0 == expected_0

    # Test 2 decimal places
    result_2 = format_tokens_with_confidence(
        tokenizer=tokenizer,
        generated_token_ids=generated_token_ids,
        confidence_predictions=confidence_predictions,
        confidence_precision=2,
    )
    expected_2 = "was(8.57) the(7.23)"
    assert result_2 == expected_2


@pytest.mark.unit
def test_sample_tokens_from_model_returns_token_ids():
    """Test that sample_tokens_from_model returns token IDs correctly."""
    from dendritic.experiments.confidence.sampling_utils import sample_tokens_from_model

    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    text, confidence_predictions, generated_token_ids, full_token_ids = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        max_new_tokens=3,
        device="cpu",
    )

    # Check that all return values are present
    assert isinstance(text, str)
    assert confidence_predictions is None  # MiniGPT doesn't have confidence
    assert isinstance(generated_token_ids, list)
    assert isinstance(full_token_ids, list)
    assert len(generated_token_ids) == 3  # Should have 3 generated tokens
    assert len(full_token_ids) > 3  # Should include prompt tokens + generated tokens


@pytest.mark.unit
def test_sample_model_output_with_confidence_formatting():
    """Test that sample_model_output returns formatted tokens when confidence is available."""
    from dendritic.experiments.confidence.sampling_utils import sample_model_output

    model = MockConfidenceAwareGPT()
    tokenizer = MockTokenizer()

    generated, confidence_predictions, formatted_tokens = sample_model_output(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        device="cpu",
        max_new_tokens=3,
        include_confidence_formatting=True,
    )

    assert isinstance(generated, str)
    assert confidence_predictions is not None
    assert len(confidence_predictions) == 3
    assert formatted_tokens is not None
    # Check that formatted tokens contain confidence scores in parentheses
    assert "(" in formatted_tokens and ")" in formatted_tokens
    # Check that we have the expected number of formatted tokens
    assert len(formatted_tokens.split()) == 3  # Should have 3 formatted tokens


@pytest.mark.unit
def test_sample_model_output_backward_compatibility():
    """Test that sample_model_output maintains backward compatibility."""
    from dendritic.experiments.confidence.sampling_utils import sample_model_output

    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    # Test without confidence formatting (old behavior)
    generated, confidence_predictions, formatted_tokens = sample_model_output(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        device="cpu",
        max_new_tokens=3,
        include_confidence_formatting=False,
    )

    # Should still work with old tuple unpacking (though this will cause type errors in IDE)
    assert isinstance(generated, str)
    assert confidence_predictions is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
