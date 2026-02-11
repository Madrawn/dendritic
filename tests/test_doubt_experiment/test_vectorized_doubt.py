"""
Unit tests for vectorized doubt in DoubtAwareGPT.

These tests verify that the doubt_vector_dim parameter is correctly
integrated into the model and that the forward pass and training work with vectorized doubt predictions.
"""

import pytest
import torch
from dendritic.experiments.models.doubt_conditioning.DoubtAwareGPT import DoubtAwareGPT
from dendritic.experiments.models.ModelConfig import ModelConfig


@pytest.fixture
def minimal_config():
    """Create a minimal ModelConfig for testing."""
    return ModelConfig(
        vocab_size=1000,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=16,
        hidden_dim=64,
        dropout=0.1,
    )


@pytest.mark.unit
def test_doubt_vector_dim_default(minimal_config):
    """Test that default doubt_vector_dim is 1 (scalar)."""
    model = DoubtAwareGPT(config=minimal_config)
    assert model.doubt_vector_dim == 1
    assert model.loss_predictor.out_features == 1


@pytest.mark.unit
def test_doubt_vector_dim_custom(minimal_config):
    """Test that custom doubt_vector_dim is set correctly."""
    model = DoubtAwareGPT(config=minimal_config, doubt_vector_dim=16)
    assert model.doubt_vector_dim == 16
    assert model.loss_predictor.out_features == 16


@pytest.mark.unit
def test_doubt_vector_dim_from_config(minimal_config):
    """Test that doubt_vector_dim is taken from config when not passed as kwarg."""
    minimal_config.doubt_vector_dim = 8
    model = DoubtAwareGPT(config=minimal_config)
    assert model.doubt_vector_dim == 8
    assert model.loss_predictor.out_features == 8


@pytest.mark.unit
def test_doubt_vector_dim_validation(minimal_config):
    """Test that invalid doubt_vector_dim raises error."""
    with pytest.raises(ValueError, match="doubt_vector_dim must be a positive integer"):
        DoubtAwareGPT(config=minimal_config, doubt_vector_dim=0)
    with pytest.raises(ValueError, match="doubt_vector_dim must be a positive integer"):
        DoubtAwareGPT(config=minimal_config, doubt_vector_dim=-1)


@pytest.mark.unit
def test_forward_output_shape_scalar(minimal_config):
    """Test forward pass with scalar doubt (default)."""
    model = DoubtAwareGPT(config=minimal_config)
    model.eval()
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids)

    assert "logits" in output
    assert "loss_prediction" in output
    assert output["logits"].shape == (batch_size, seq_len, minimal_config.vocab_size)
    # With default doubt_vector_dim=1, shape is [B, T, 1]
    assert output["loss_prediction"].shape == (batch_size, seq_len, 1)


@pytest.mark.unit
def test_forward_output_shape_vector(minimal_config):
    """Test forward pass with vectorized doubt."""
    V = 16
    model = DoubtAwareGPT(config=minimal_config, doubt_vector_dim=V)
    model.eval()
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids)

    assert output["loss_prediction"].shape == (batch_size, seq_len, V)


@pytest.mark.unit
def test_doubt_scalar_shape_forward(minimal_config):
    """Test that providing doubt_scalars with correct shape works."""
    V = 8
    model = DoubtAwareGPT(config=minimal_config, doubt_vector_dim=V)
    model.eval()
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len))
    doubt_scalars = torch.rand(batch_size, seq_len, V)

    with torch.no_grad():
        output = model(input_ids, doubt_scalars=doubt_scalars)

    assert "logits" in output
    assert output["logits"].shape == (batch_size, seq_len, minimal_config.vocab_size)


@pytest.mark.unit
def test_doubt_scalar_shape_mismatch_raises(minimal_config):
    """Test that providing doubt_scalars with wrong shape raises error."""
    V = 8
    model = DoubtAwareGPT(config=minimal_config, doubt_vector_dim=V)
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len))
    # Wrong shape: last dim should be V but is 1
    doubt_scalars = torch.rand(batch_size, seq_len, 1)

    with pytest.raises(RuntimeError):
        # The model will try to broadcast or fail in the linear layer
        model(input_ids, doubt_scalars=doubt_scalars)


@pytest.mark.unit
def test_adaptive_layer_vector_input(minimal_config):
    """Test that AdaptiveLayer correctly processes vector doubt."""
    from dendritic.experiments.models.doubt_conditioning.MetaAwareBlock import AdaptiveLayer

    dim = 32
    V = 16
    layer = AdaptiveLayer(dim, doubt_vector_dim=V)
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, dim)
    doubt_scalar = torch.rand(batch_size, seq_len, V)

    output = layer(x, doubt_scalar)
    assert output.shape == x.shape


@pytest.mark.unit
def test_adaptive_layer_scalar_input(minimal_config):
    """Test that AdaptiveLayer works with scalar doubt (V=1)."""
    from dendritic.experiments.models.doubt_conditioning.MetaAwareBlock import AdaptiveLayer

    dim = 32
    layer = AdaptiveLayer(dim, doubt_vector_dim=1)
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, dim)
    doubt_scalar = torch.rand(batch_size, seq_len, 1)

    output = layer(x, doubt_scalar)
    assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
