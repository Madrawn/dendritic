from dendritic.layers.DendriticStack import DendriticStack


import pytest
import torch
# Fixtures for common test configurations
@pytest.fixture(params=[(64, 32, 8), (128, 64, 16), (256, 128, 32)])
def layer_config(request):
    input_dim, output_dim, poly_rank = request.param
    return input_dim, output_dim, poly_rank

@pytest.fixture
def dendritic_stack(layer_config):
    input_dim, output_dim, poly_rank = layer_config
    return DendriticStack(input_dim, output_dim, poly_rank=poly_rank)
# ===================
# DendriticStack Tests
# ===================
class TestDendriticStack:
    @pytest.mark.unit
    def test_forward_pass_shape(self, dendritic_stack, layer_config):
        input_dim, output_dim, _ = layer_config
        x = torch.randn(16, input_dim)
        y = dendritic_stack(x)
        assert y.shape == (16, output_dim)

    @pytest.mark.unit
    def test_diagonal_pathway(self):
        # Test with diagonal pathway enabled
        layer_with_diag = DendriticStack(64, 32, poly_rank=8, diag_rank=4)
        layer_no_diag = DendriticStack(64, 32, poly_rank=8, diag_rank=0)

        x = torch.randn(16, 64)
        y_diag = layer_with_diag(x)
        y_no_diag = layer_no_diag(x)

        assert not torch.allclose(y_diag, y_no_diag)

    @pytest.mark.unit
    def test_bias_handling(self):
        layer_with_bias = DendriticStack(64, 32, bias=True)
        layer_no_bias = DendriticStack(64, 32, bias=False)

        x = torch.randn(16, 64)
        y_bias = layer_with_bias(x)
        y_no_bias = layer_no_bias(x)

        assert not torch.allclose(y_bias, y_no_bias)

    @pytest.mark.unit
    def test_independent_inputs_flag(self):
        # When independent_inputs=True, diag_rank should equal poly_rank
        layer_independent = DendriticStack(64, 32, poly_rank=8, independent_inputs=True)
        layer_dependent = DendriticStack(64, 32, poly_rank=8, independent_inputs=False)

        # Check diag_rank attribute
        assert layer_independent.diag_rank == 8
        assert layer_dependent.diag_rank == max(4, 8 // 4)  # max(4, 2) = 4

        # Ensure diagonal pathway is present
        assert layer_independent.use_diagonal == True
        assert layer_dependent.use_diagonal == True

    @pytest.mark.unit
    def test_parameter_count_consistency(self):
        # Test that parameter_count matches actual number of parameters
        input_dim, output_dim, poly_rank = 64, 32, 8
        layer = DendriticStack(input_dim, output_dim, poly_rank=poly_rank)
        actual_params = sum(p.numel() for p in layer.parameters())
        expected_params = DendriticStack.parameter_count(
            input_dim, output_dim, poly_rank=poly_rank, include_linear=True
        )
        assert actual_params == expected_params

        # Test with include_linear=False (should exclude linear parameters)
        expected_without_linear = DendriticStack.parameter_count(
            input_dim, output_dim, poly_rank=poly_rank, include_linear=False
        )
        linear_params = sum(p.numel() for p in layer.linear.parameters())
        assert actual_params - linear_params == expected_without_linear

    @pytest.mark.unit
    def test_poly_degree(self):
        # Test different poly_degree values
        for degree in [1, 2, 3, 4]:
            layer = DendriticStack(64, 32, poly_rank=8, poly_degree=degree)
            x = torch.randn(16, 64)
            y = layer(x)
            assert y.shape == (16, 32)