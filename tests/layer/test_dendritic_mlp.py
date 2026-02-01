from dendritic.layers.DendriticMLP import DendriticMLP


import pytest
import torch
import torch.nn as nn


# Fixtures for common test configurations
@pytest.fixture(params=[(64, 32, 8), (128, 64, 16), (256, 128, 32)])
def layer_config(request):
    input_dim, output_dim, poly_rank = request.param
    return input_dim, output_dim, poly_rank


@pytest.fixture
def dendritic_mlp():
    return DendriticMLP(256, 1024, poly_rank=16)


# ===================
# DendriticMLP Tests
# ===================
class TestDendriticMLP:
    @pytest.mark.unit
    def test_forward_pass_shape(self, dendritic_mlp):
        # Test transformer-like shape
        x = torch.randn(4, 128, 256)  # [batch, seq, embed]
        y = dendritic_mlp(x)
        assert y.shape == (4, 128, 256)

    @pytest.mark.unit
    def test_activation_functions(self):
        activations = [nn.ReLU(), nn.GELU(), nn.SiLU()]
        for activation in activations:
            mlp = DendriticMLP(256, 1024, activation=activation)
            x = torch.randn(4, 128, 256)
            y = mlp(x)
            assert y.shape == (4, 128, 256)

    @pytest.mark.unit
    def test_dropout(self):
        mlp = DendriticMLP(256, 1024, dropout=0.5)
        x = torch.randn(4, 128, 256)

        # Test training mode
        mlp.train()
        y_train = mlp(x)

        # Test eval mode
        mlp.eval()
        y_eval = mlp(x)

        # Dropout should create differences between train and eval outputs
        assert not torch.allclose(y_train, y_eval)

    @pytest.mark.unit
    def test_gate_mechanism(self):
        # Test compatibility with gate mechanisms
        mlp = DendriticMLP(256, 1024)
        x = torch.randn(4, 128, 256)
        gate = torch.sigmoid(torch.randn(4, 128, 256))

        # Apply gate
        y_gated = mlp(x) * gate
        y_ungated = mlp(x)

        assert not torch.allclose(y_gated, y_ungated)
