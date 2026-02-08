import pytest
import torch
from dendritic.layers.DendriticLayer import DendriticLayer


# =====================
# Tensor Input Edge Cases
# =====================
@pytest.mark.xfail
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),  # Minimum dimensions
        (1, 10000),  # Very wide tensor
        (10000, 1),  # Very tall tensor
        (0, 64),  # Empty batch
        (16, 0),  # Empty features
    ],
)
@pytest.mark.edge
def test_tensor_shape_edge_cases(shape):
    """Test extreme tensor shapes in DendriticLayer"""
    input_dim = shape[1] if shape[1] > 0 else 1
    model = DendriticLayer(input_dim, 32, poly_rank=8)

    if 0 in shape:
        with pytest.raises(RuntimeError):
            x = torch.randn(*shape)
            model(x)
    else:
        x = torch.randn(*shape)
        y = model(x)
        assert y.shape == (shape[0], 32)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "value",
    [
        torch.finfo(torch.float32).max,
        torch.finfo(torch.float32).min,
        torch.finfo(torch.float32).eps,
        0.0,
        -0.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
@pytest.mark.edge
def test_numerical_edge_cases(value):
    """Test numerical stability with extreme values"""
    model = DendriticLayer(64, 32, poly_rank=8)
    x = torch.full((16, 64), value)
    y = model(x)

    # Check for NaN/Inf in outputs
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


# =====================
# Model Architecture Edge Cases
# =====================


@pytest.mark.parametrize(
    "config",
    [
        {"input_dim": 1, "output_dim": 1, "poly_rank": 1},  # Minimal config
        {"input_dim": 10000, "output_dim": 1, "poly_rank": 1},  # Extreme input
        {"input_dim": 1, "output_dim": 10000, "poly_rank": 1},  # Extreme output
        {"input_dim": 512, "output_dim": 512, "poly_rank": 512},  # Max rank
    ],
)
@pytest.mark.edge
def test_model_size_edge_cases(config):
    """Test boundary model architectures"""
    if config["poly_rank"] <= 0:
        with pytest.raises(ValueError):
            DendriticLayer(**config)
    else:
        model = DendriticLayer(**config)
        x = torch.randn(16, config["input_dim"])
        y = model(x)
        assert y.shape == (16, config["output_dim"])


# =====================
# Sequence Length Edge Cases
# =====================


@pytest.mark.parametrize("seq_len", [1, 2**9, 2**10, 2**11])
@pytest.mark.edge
def test_sequence_length_boundaries(seq_len):
    """Test extreme sequence lengths in models"""
    model = DendriticLayer(64, 32, poly_rank=8)
    x = torch.randn(16, seq_len, 64)
    y = model(x)

    assert y.shape == (16, seq_len, 32)
    assert not torch.isnan(y).any()
