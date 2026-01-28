from dendritic.layers.DendriticLayer import DendriticLayer


import pytest
import torch


@pytest.fixture
def dendritic_layer(layer_config):
    input_dim, output_dim, poly_rank = layer_config
    return DendriticLayer(input_dim, output_dim, poly_rank=poly_rank)


# Fixtures for common test configurations
@pytest.fixture(params=[(64, 32, 8), (128, 64, 16), (256, 128, 32)])
def layer_config(request):
    input_dim, output_dim, poly_rank = request.param
    return input_dim, output_dim, poly_rank


# ===================
# DendriticLayer Tests
# ===================
class TestDendriticLayer:
    @pytest.mark.unit
    def test_forward_pass_shape(self, dendritic_layer, layer_config):
        input_dim, output_dim, _ = layer_config
        x = torch.randn(16, input_dim)
        y = dendritic_layer(x)
        assert y.shape == (16, output_dim)

    @pytest.mark.unit
    def test_3d_forward_pass(self, dendritic_layer, layer_config):
        input_dim, output_dim, _ = layer_config
        x = torch.randn(4, 8, input_dim)  # [batch, seq, features]
        y = dendritic_layer(x)
        assert y.shape == (4, 8, output_dim)

    @pytest.mark.unit
    def test_gradient_flow(self, dendritic_layer, layer_config):
        input_dim, _, _ = layer_config
        x = torch.randn(16, input_dim, requires_grad=True)
        y = dendritic_layer(x)
        loss = y.sum()
        loss.backward()

        # Verify gradients exist for all parameters
        for param in dendritic_layer.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    @pytest.mark.unit
    def test_weight_initialization(self, dendritic_layer):
        # Check weights are initialized properly
        for name, param in dendritic_layer.named_parameters():
            if "weight" in name:
                assert not torch.allclose(param, torch.zeros_like(param))
                assert not torch.isnan(param).any()

    @pytest.mark.unit
    def test_diagonal_pathway(self):
        # Test with diagonal pathway enabled
        layer_with_diag = DendriticLayer(64, 32, poly_rank=8, diag_rank=4)
        layer_no_diag = DendriticLayer(64, 32, poly_rank=8, diag_rank=0)

        x = torch.randn(16, 64)
        y_diag = layer_with_diag(x)
        y_no_diag = layer_no_diag(x)

        assert not torch.allclose(y_diag, y_no_diag)

    @pytest.mark.unit
    def test_bias_handling(self):
        layer_with_bias = DendriticLayer(64, 32, bias=True)
        layer_no_bias = DendriticLayer(64, 32, bias=False)

        x = torch.randn(16, 64)
        y_bias = layer_with_bias(x)
        y_no_bias = layer_no_bias(x)

        assert not torch.allclose(y_bias, y_no_bias)

    @pytest.mark.unit
    def test_edge_cases(self, dendritic_layer):
        # Zero input
        x_zero = torch.zeros(16, dendritic_layer.input_dim)
        y_zero = dendritic_layer(x_zero)
        assert not torch.isnan(y_zero).any()

        # Large values
        x_large = torch.randn(16, dendritic_layer.input_dim) * 100
        y_large = dendritic_layer(x_large)
        assert not torch.isnan(y_large).any()

    @pytest.mark.unit
    def test_mathematical_correctness(self):
        # Test quadratic computation accuracy
        # Disable diagonal pathway explicitly for this test
        layer = DendriticLayer(2, 1, poly_rank=1, diag_rank=0)

        # Manually set weights to known values
        with torch.no_grad():
            # Use copy_ to modify the underlying projections tensor
            layer.w1.data.copy_(torch.tensor([[1.0, 0.0]]))
            layer.w2.data.copy_(torch.tensor([[0.0, 1.0]]))

            # Zero out the biases (reset_parameters fills them with small noise)
            layer.proj_biases.data.zero_()

            layer.poly_out.data.copy_(torch.tensor([[1.0]]))
            layer.linear.weight.data.zero_()
            layer.linear.bias.data.zero_()

            # Note: alpha_grad_boost is 1e-3, so if you want
            # an effective alpha of 1.0, set alpha to 0.999
            # OR just account for the 1.001 multiplier in the assertion.
            layer.alpha.data.fill_(1.0 - layer.alpha_grad_boost)
        x = torch.tensor([[2.0, 3.0]])
        y = layer(x)
        assert torch.allclose(y, torch.tensor([[6.0]]), atol=1e-4)  # 2*3 = 6

    @pytest.mark.unit
    def test_diag_rank_validation(self):
        # Valid diag_rank values should not raise
        DendriticLayer(64, 32, diag_rank=0)
        DendriticLayer(64, 32, diag_rank=10)
        DendriticLayer(64, 32, diag_rank="auto")
        # Invalid diag_rank should raise ValueError
        with pytest.raises(ValueError):
            DendriticLayer(64, 32, diag_rank=-1)
