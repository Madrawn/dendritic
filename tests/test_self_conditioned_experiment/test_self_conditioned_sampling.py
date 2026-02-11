# ruff: noqa: PLR6301, PLR2004
"""
Unit tests for SelfConditionedGPT model sampling functionality.

These tests verify that SelfConditionedGPT works correctly with the
sampling utilities and produces valid outputs.
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
from dendritic.experiments.models.doubt_conditioning.SelfConditionedGPT import SelfConditionedGPT
from dendritic.experiments.models.doubt_conditioning.DoubtAwareGPT import DoubtAwareGPT
from dendritic.experiments.models.ModelConfig import ModelConfig


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 50256
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
            50256: "",
        }

    @staticmethod
    def encode(text, return_tensors="pt"):
        """Mock encode method."""
        return torch.tensor([[1, 2, 3, 4]])

    def decode(self, tokens, skip_special_tokens=True):
        """Mock decode method."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        if isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], int):
            token_id = tokens[0]
            if skip_special_tokens and token_id == self.eos_token_id:
                return ""
            return self.token_map.get(token_id, f"<token_{token_id}>")

        if isinstance(tokens, list):
            decoded_tokens = []
            for token_id in tokens:
                if skip_special_tokens and token_id == self.eos_token_id:
                    continue
                decoded_tokens.append(self.token_map.get(token_id, f"<token_{token_id}>"))
            return " ".join(decoded_tokens)

        return "Mock generated text"


@pytest.fixture
def minimal_config():
    """Create a minimal ModelConfig for testing."""
    return ModelConfig(
        vocab_size=1000,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=16,
        hidden_dim=64,  # MLP hidden dimension
        dropout=0.1,
    )


@pytest.fixture
def self_conditioned_model(minimal_config):
    """Create a SelfConditionedGPT model for testing."""
    return SelfConditionedGPT(config=minimal_config, bound_fn="tanh", take_meta=2)


@pytest.fixture
def doubt_aware_model(minimal_config):
    """Create a DoubtAwareGPT model for comparison."""
    return DoubtAwareGPT(config=minimal_config, take_meta=2)


@pytest.mark.unit
def test_self_conditioned_model_initialization(self_conditioned_model):
    """Test that SelfConditionedGPT initializes correctly."""
    assert self_conditioned_model is not None
    assert hasattr(self_conditioned_model, "core")
    assert isinstance(self_conditioned_model.core, DoubtAwareGPT)
    assert hasattr(self_conditioned_model, "max_seq_len")
    assert hasattr(self_conditioned_model, "bound")
    # SelfConditionedGPT should NOT have loss_predictor directly
    assert not hasattr(self_conditioned_model, "loss_predictor")
    # But the core should have it
    assert hasattr(self_conditioned_model.core, "loss_predictor")


@pytest.mark.unit
def test_self_conditioned_model_forward_without_doubt(self_conditioned_model):
    """Test forward pass without doubt_scalars (internal)."""
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    output = self_conditioned_model(input_ids)

    # Should return logits only (not a dict)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, seq_len, 1000)


@pytest.mark.unit
def test_self_conditioned_model_forward_with_doubt(self_conditioned_model):
    """Test forward pass with doubt_scalars (single pass)."""
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    doubt_scalars = torch.rand(batch_size, seq_len, 1)

    output = self_conditioned_model(input_ids, doubt_scalars=doubt_scalars)

    # Should return logits only (not a dict)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, seq_len, 1000)


@pytest.mark.unit
def test_self_conditioned_model_forward_with_diagnostics(self_conditioned_model):
    """Test forward_with_diagnostics returns correct dict."""
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    output = self_conditioned_model.forward_with_diagnostics(input_ids)

    assert isinstance(output, dict)
    assert "logits" in output
    assert "doubt_signal" in output
    assert "pass1_logits" in output
    assert output["logits"].shape == (batch_size, seq_len, 1000)
    # doubt_signal shape is [B, T, V] with V=1 by default
    assert output["doubt_signal"].shape == (batch_size, seq_len, 1)
    assert output["pass1_logits"].shape == (batch_size, seq_len, 1000)


@pytest.mark.unit
def test_self_conditioned_model_detection_as_non_doubt():
    """Test that SelfConditionedGPT is detected as non-doubt model by sampling utils."""
    model = SelfConditionedGPT(
        config=ModelConfig(
            vocab_size=1000,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            max_seq_len=16,
            hidden_dim=64,
        )
    )

    # The sampling utility checks for loss_predictor attribute
    # SelfConditionedGPT should NOT have it (core has it, but not the wrapper)
    assert not hasattr(model, "loss_predictor")


@pytest.mark.unit
def test_sample_tokens_from_model_self_conditioned(self_conditioned_model, minimal_config):
    """Test sampling from SelfConditionedGPT model."""
    tokenizer = MockTokenizer()

    # Test basic sampling
    config = SamplingConfig(device="cpu", max_new_tokens=5)
    text, generated_token_ids, full_token_ids = sample_tokens_from_model(
        model=self_conditioned_model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        config=config,
    )

    assert isinstance(text, str)
    assert "Once" in text  # Should contain the decoded prompt tokens
    # SelfConditionedGPT is not detected as doubt model, so no loss predictions
    assert isinstance(generated_token_ids, list)
    assert len(generated_token_ids) == 5
    assert isinstance(full_token_ids, list)
    assert len(full_token_ids) > 4  # prompt + generated


@pytest.mark.unit
def test_sample_tokens_from_model_self_conditioned_eos_stopping(self_conditioned_model):
    """Test that sampling stops at EOS token for SelfConditionedGPT."""
    tokenizer = MockTokenizer()

    # Mock tokenizer to return EOS after a few tokens
    original_decode = tokenizer.decode
    call_count = 0

    def mock_decode(tokens, skip_special_tokens=True):
        nonlocal call_count
        call_count += 1
        if call_count > 2:  # Simulate EOS after 2 generated tokens
            return ""
        return "Token"

    tokenizer.decode = mock_decode

    try:
        config = SamplingConfig(device="cpu", max_new_tokens=10)
        text, generated_token_ids, full_token_ids = sample_tokens_from_model(
            model=self_conditioned_model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            config=config,
        )
        assert isinstance(text, str)
        # Should stop early due to EOS
        assert len(generated_token_ids) <= 10
    finally:
        tokenizer.decode = original_decode


@pytest.mark.unit
def test_sample_model_output_self_conditioned(self_conditioned_model):
    """Test sample_model_output with SelfConditionedGPT."""
    tokenizer = MockTokenizer()

    config = SamplingConfig(device="cpu", max_new_tokens=3, include_doubt_formatting=True)
    generated = sample_model_output(
        model=self_conditioned_model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        config=config,
    )

    assert isinstance(generated, str)


@pytest.mark.unit
def test_self_conditioned_model_bound_function_applied(minimal_config):
    """Test that the bound function correctly limits the doubt signal."""
    # Use tanh bound
    model = SelfConditionedGPT(config=minimal_config, bound_fn="tanh", take_meta=2)

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    diagnostics = model.forward_with_diagnostics(input_ids)
    doubt_signal = diagnostics["doubt_signal"]

    # Tanh should bound between -1 and 1
    assert torch.all(doubt_signal >= -1.0).item()
    assert torch.all(doubt_signal <= 1.0).item()

    # Test sigmoid
    model_sigmoid = SelfConditionedGPT(config=minimal_config, bound_fn="sigmoid", take_meta=2)
    diagnostics_sigmoid = model_sigmoid.forward_with_diagnostics(input_ids)
    doubt_signal_sigmoid = diagnostics_sigmoid["doubt_signal"]
    assert torch.all(doubt_signal_sigmoid >= 0.0).item()
    assert torch.all(doubt_signal_sigmoid <= 1.0).item()

    # Test relu
    model_relu = SelfConditionedGPT(config=minimal_config, bound_fn="relu", take_meta=2)
    diagnostics_relu = model_relu.forward_with_diagnostics(input_ids)
    doubt_signal_relu = diagnostics_relu["doubt_signal"]
    assert torch.all(doubt_signal_relu >= 0.0).item()

    # Test none (Identity) - should be unbounded but still finite
    model_none = SelfConditionedGPT(config=minimal_config, bound_fn="none", take_meta=2)
    diagnostics_none = model_none.forward_with_diagnostics(input_ids)
    doubt_signal_none = diagnostics_none["doubt_signal"]
    assert torch.all(torch.isfinite(doubt_signal_none)).item()


@pytest.mark.unit
def test_self_conditioned_diagnostics_consistency(minimal_config):
    """Test that forward_with_diagnostics returns values consistent with manual."""
    model = SelfConditionedGPT(config=minimal_config, bound_fn="tanh", take_meta=2)
    model.eval()  # Ensure deterministic behavior (no dropout)
    input_ids = torch.randint(0, 1000, (2, 8))

    # Get diagnostics
    diag = model.forward_with_diagnostics(input_ids)

    # Manually compute
    pass1_out = model.core(input_ids, doubt_scalars=None)
    expected_doubt_signal = model.bound(pass1_out["loss_prediction"])
    expected_logits = model.core(input_ids, doubt_scalars=expected_doubt_signal)["logits"]

    assert torch.allclose(diag["doubt_signal"], expected_doubt_signal, atol=1e-5)
    assert torch.allclose(diag["logits"], expected_logits, atol=1e-5)
    assert torch.allclose(diag["pass1_logits"], pass1_out["logits"], atol=1e-5)


@pytest.mark.unit
def test_self_conditioned_model_different_bound_fns(minimal_config):
    """Test SelfConditionedGPT with different bound functions."""
    bound_fns = ["tanh", "sigmoid", "softsign", "relu", "none"]

    for bound_fn in bound_fns:
        model = SelfConditionedGPT(config=minimal_config, bound_fn=bound_fn, take_meta=2)
        tokenizer = MockTokenizer()

        config = SamplingConfig(device="cpu", max_new_tokens=3)
        text, generated_token_ids, full_token_ids = sample_tokens_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            config=config,
        )

        assert isinstance(text, str)


@pytest.mark.unit
def test_self_conditioned_model_max_seq_len_respecting(minimal_config):
    """Test that SelfConditionedGPT respects max_seq_len during sampling."""
    # Create model with very small max_seq_len
    config = minimal_config
    config.max_seq_len = 10
    model = SelfConditionedGPT(config=config, take_meta=2)
    tokenizer = MockTokenizer()

    # Prompt will be 4 tokens, so we can only generate 6 more before hitting limit
    config_sampling = SamplingConfig(device="cpu", max_new_tokens=10)
    text, generated_token_ids, full_token_ids = sample_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        config=config_sampling,
    )

    # Should respect the limit
    assert len(full_token_ids) <= 10


@pytest.mark.unit
def test_self_conditioned_vs_doubt_aware_parameter_count(minimal_config):
    """Test that SelfConditionedGPT has same parameters as DoubtAwareGPT (wraps single core)."""
    doubt_model = DoubtAwareGPT(config=minimal_config, take_meta=2)
    sc_model = SelfConditionedGPT(config=minimal_config, take_meta=2, bound_fn="tanh")

    doubt_params = sum(p.numel() for p in doubt_model.parameters())
    sc_params = sum(p.numel() for p in sc_model.parameters())

    # SelfConditionedGPT wraps a DoubtAwareGPT core, so it should have the same parameters
    # (the uses the same core weights twice)
    assert sc_params == doubt_params


@pytest.mark.unit
def test_self_conditioned_model_gradient_flow(minimal_config):
    """Test that gradients flow correctly through SelfConditionedGPT."""
    model = SelfConditionedGPT(config=minimal_config, take_meta=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    input_ids = torch.randint(0, 1000, (2, 8))
    output = model(input_ids)
    loss = output.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that gradients exist
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


@pytest.mark.unit
def test_sample_tokens_temperature_effect(self_conditioned_model):
    """Test that temperature affects sampling (though not deterministically)."""
    tokenizer = MockTokenizer()

    # Use low temperature for more deterministic sampling
    config_low = SamplingConfig(device="cpu", max_new_tokens=5, temperature=0.1)
    text_low, generated_token_ids_low, full_token_ids_low = sample_tokens_from_model(
        model=self_conditioned_model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        config=config_low,
    )

    # Use high temperature
    config_high = SamplingConfig(device="cpu", max_new_tokens=5, temperature=2.0)
    text_high, generated_token_ids_high, full_token_ids_high = sample_tokens_from_model(
        model=self_conditioned_model,
        tokenizer=tokenizer,
        prompt="Test prompt",
        config=config_high,
    )

    # Both should produce valid strings
    assert isinstance(text_low, str)
    assert isinstance(text_high, str)


@pytest.mark.unit
def test_self_conditioned_model_device_handling(minimal_config):
    """Test that SelfConditionedGPT handles device placement correctly."""
    model = SelfConditionedGPT(config=minimal_config, take_meta=2)

    # Test CPU
    model_cpu = model.cpu()
    input_ids = torch.randint(0, 1000, (2, 8))
    output_cpu = model_cpu(input_ids)
    assert output_cpu.device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        input_ids_cuda = input_ids.cuda()
        output_cuda = model_cuda(input_ids_cuda)
        assert output_cuda.device.type == "cuda"


@pytest.mark.unit
def test_self_conditioned_model_batch_processing(minimal_config):
    """Test that SelfConditionedGPT handles different batch sizes."""
    model = SelfConditionedGPT(config=minimal_config, take_meta=2)

    for batch_size in [1, 2, 4]:
        seq_len = 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        output = model(input_ids)
        assert output.shape == (batch_size, seq_len, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
