# ruff: noqa: PLR6301, PLR2004

"""
Tests for token sampling functionality in doubt experiments.
"""

import pytest
import torch
from torch import nn
from unittest.mock import patch
from dendritic.experiments.doubt.sampling_utils import (
    sample_tokens_from_model,
    sample_model_output,
    SamplingConfig,
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

    @staticmethod
    def encode(text, return_tensors="pt"):
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

    def forward(self, input_ids, *args, **kwargs) -> dict[str, torch.Tensor] | torch.Tensor:
        """Mock forward method."""
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, self.vocab_size)


class MockDoubtAwareGPT(MockMiniGPT):
    """Mock DoubtAwareGPT model for testing."""

    def __init__(self):
        super().__init__()
        self.loss_predictor = nn.Linear(10, 1)
        self.doubt_vector_dim = 1  # Default scalar

    def forward(self, input_ids, doubt_scalars=None) -> dict[str, torch.Tensor] | torch.Tensor:
        """Mock forward method."""
        batch_size, seq_len = input_ids.shape
        V = self.doubt_vector_dim
        return {
            "logits": torch.randn(batch_size, seq_len, self.vocab_size),
            "loss_prediction": torch.randn(batch_size, seq_len, V),
        }


@pytest.mark.unit
def test_sample_tokens_from_model_minigpt():
    """Test sampling from MiniGPT model."""
    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    # Test basic sampling
    config = SamplingConfig(device="cpu", max_new_tokens=5)
    text, generated_token_ids, full_token_ids = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        config=config,
    )

    assert isinstance(text, str)
    assert "Once" in text  # Should contain the decoded prompt tokens


@pytest.mark.unit
def test_sample_tokens_from_model_top_p_sampling():
    """Test top-p (nucleus) sampling."""
    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    # Test with top_p < 1.0
    config = SamplingConfig(device="cpu", max_new_tokens=5, temperature=0.8, top_p=0.9)
    text, generated_token_ids, full_token_ids = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        config=config,
    )

    assert isinstance(text, str)


@pytest.mark.unit
def test_sample_tokens_from_model_temperature():
    """Test temperature scaling."""
    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    # Test with different temperatures
    config_low = SamplingConfig(device="cpu", max_new_tokens=5, temperature=0.5)
    text_low_temp, generated_token_ids_low, full_token_ids_low = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        config=config_low,
    )

    config_high = SamplingConfig(device="cpu", max_new_tokens=5, temperature=1.5)
    text_high_temp, generated_token_ids_high, full_token_ids_high = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        config=config_high,
    )

    assert isinstance(text_low_temp, str)
    assert isinstance(text_high_temp, str)


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
        config = SamplingConfig(device="cpu", max_new_tokens=10)
        text, generated_token_ids, full_token_ids = sample_tokens_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            config=config,
        )
        assert isinstance(text, str)
    finally:
        tokenizer.decode = original_decode


@pytest.mark.unit
def test_format_tokens_with_doubt():
    """Test the format_tokens_with_doubt function."""
    from dendritic.experiments.doubt.sampling_utils import format_tokens_with_doubt

    tokenizer = MockTokenizer()

    # Test basic formatting
    generated_token_ids = [5, 6, 7]  # "was", "the", "day"
    loss_predictions = [8.5, 7.2, 6.8]

    result = format_tokens_with_doubt(
        tokenizer=tokenizer,
        generated_token_ids=generated_token_ids,
        loss_predictions=loss_predictions,
        doubt_precision=1,
    )

    expected = "was(8.5) the(7.2) day(6.8)"
    assert result == expected


@pytest.mark.unit
def test_format_tokens_with_doubt_whitespace_stripping():
    """Test that whitespace is properly stripped from decoded tokens."""
    from dendritic.experiments.doubt.sampling_utils import format_tokens_with_doubt

    class WhitespaceTokenizer(MockTokenizer):
        def decode(self, tokens, skip_special_tokens=True):
            # Add leading space to simulate some tokenizers
            if isinstance(tokens, list) and len(tokens) == 1:
                return f" {self.token_map.get(tokens[0], f'<token_{tokens[0]}>')}"
            return super().decode(tokens, skip_special_tokens)

    tokenizer = WhitespaceTokenizer()
    generated_token_ids = [5, 6, 7]  # " was", " the", " day"
    loss_predictions = [8.5, 7.2, 6.8]

    result = format_tokens_with_doubt(
        tokenizer=tokenizer,
        generated_token_ids=generated_token_ids,
        loss_predictions=loss_predictions,
        doubt_precision=1,
    )

    expected = "was(8.5) the(7.2) day(6.8)"  # Should be stripped of leading spaces
    assert result == expected


@pytest.mark.unit
def test_format_tokens_with_doubt_validation():
    """Test that format_tokens_with_doubt validates token count match."""
    from dendritic.experiments.doubt.sampling_utils import format_tokens_with_doubt

    tokenizer = MockTokenizer()

    # Test mismatched lengths
    generated_token_ids = [5, 6]  # 2 tokens
    loss_predictions = [8.5]  # 1 loss value

    with pytest.raises(ValueError, match="Token count .* doesn't match loss prediction count"):
        format_tokens_with_doubt(
            tokenizer=tokenizer,
            generated_token_ids=generated_token_ids,
            loss_predictions=loss_predictions,
        )


@pytest.mark.unit
def test_format_tokens_with_doubt_precision():
    """Test different precision levels for doubt scores."""
    from dendritic.experiments.doubt.sampling_utils import format_tokens_with_doubt

    tokenizer = MockTokenizer()
    generated_token_ids = [5, 6]
    loss_predictions = [8.56789, 7.23456]

    # Test 0 decimal places
    result_0 = format_tokens_with_doubt(
        tokenizer=tokenizer,
        generated_token_ids=generated_token_ids,
        loss_predictions=loss_predictions,
        doubt_precision=0,
    )
    expected_0 = "was(9) the(7)"
    assert result_0 == expected_0

    # Test 2 decimal places
    result_2 = format_tokens_with_doubt(
        tokenizer=tokenizer,
        generated_token_ids=generated_token_ids,
        loss_predictions=loss_predictions,
        doubt_precision=2,
    )
    expected_2 = "was(8.57) the(7.23)"
    assert result_2 == expected_2


@pytest.mark.unit
def test_sample_tokens_from_model_returns_token_ids():
    """Test that sample_tokens_from_model returns token IDs correctly."""
    from dendritic.experiments.doubt.sampling_utils import sample_tokens_from_model

    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    config = SamplingConfig(device="cpu", max_new_tokens=3)
    text, generated_token_ids, full_token_ids = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        config=config,
    )

    # Check that all return values are present
    assert isinstance(text, str)
    assert isinstance(generated_token_ids, list)
    assert isinstance(full_token_ids, list)
    assert len(generated_token_ids) == 3  # Should have 3 generated tokens
    assert len(full_token_ids) > 3  # Should include prompt tokens + generated tokens


@pytest.mark.unit
def test_sample_model_output_backward_compatibility():
    """Test that sample_model_output maintains backward compatibility."""

    model = MockMiniGPT()
    tokenizer = MockTokenizer()

    # Test without doubt formatting (old behavior)
    config = SamplingConfig(device="cpu", max_new_tokens=3, include_doubt_formatting=False)
    generated = sample_model_output(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        config=config,
    )

    # Should still work with old tuple unpacking (though this will cause type errors in IDE)
    assert isinstance(generated, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
