import torch
import torch.nn.functional as F
import pytest
import numpy as np
from dendritic.layers.DendriticLayer import DendriticLayer
from dendritic.layers.DendriticMLP import DendriticMLP
from dendritic.layers.DendriticStack import DendriticStack


# Fixtures for common test configurations
@pytest.fixture(params=[(64, 32, 8), (128, 64, 16), (256, 128, 32)])
def layer_config(request):
    input_dim, output_dim, poly_rank = request.param
    return input_dim, output_dim, poly_rank


@pytest.fixture
def dendritic_stack(layer_config):
    input_dim, output_dim, poly_rank = layer_config
    return DendriticStack(input_dim, output_dim, poly_rank=poly_rank)


@pytest.fixture
def dendritic_mlp():
    return DendriticMLP(256, 1024, poly_rank=16)


@pytest.fixture
def dendritic_layer(layer_config):
    input_dim, output_dim, poly_rank = layer_config
    return DendriticLayer(input_dim, output_dim, poly_rank=poly_rank)


# ========================
# Integration & Edge Tests
# ========================
class TestIntegration:
    @pytest.mark.unit
    def test_serialization(self, dendritic_layer, layer_config):
        # Test saving and loading
        torch.save(dendritic_layer.state_dict(), "test_layer.pth")
        input_dim, output_dim, poly_rank = layer_config
        new_layer = DendriticLayer(input_dim, output_dim, poly_rank=poly_rank)
        new_layer.load_state_dict(torch.load("test_layer.pth"))

        x = torch.randn(16, dendritic_layer.input_dim)
        y_orig = dendritic_layer(x)
        y_new = new_layer(x)
        assert torch.allclose(y_orig, y_new)

    @pytest.mark.unit
    def test_device_placement(self, dendritic_layer):
        # Test CPU
        x_cpu = torch.randn(16, dendritic_layer.input_dim)
        y_cpu = dendritic_layer(x_cpu)

        # Test CUDA if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"\n[DEBUG] Moving layer to {device}")
            layer_cuda = dendritic_layer.to(device)
            print(
                f"[DEBUG] Layer device after move: {next(layer_cuda.parameters()).device}"
            )

            print(f"[DEBUG] Moving input to {device}")
            x_cuda = x_cpu.to(device)
            print(f"[DEBUG] Input device: {x_cuda.device}")

            print("[DEBUG] Running forward pass on GPU...")
            y_cuda = layer_cuda(x_cuda)
            print(f"[DEBUG] Output device: {y_cuda.device}")

            print("[DEBUG] Comparing CPU and GPU outputs...")
            cpu_numpy = y_cpu.detach().numpy()
            gpu_numpy = y_cuda.cpu().detach().numpy()

            # Print max difference for debugging
            max_diff = np.max(np.abs(cpu_numpy - gpu_numpy))
            print(f"[DEBUG] Max difference: {max_diff}")

            # Print sample values for comparison
            print(f"[DEBUG] Sample CPU value: {cpu_numpy[0, :5]}")
            print(f"[DEBUG] Sample GPU value: {gpu_numpy[0, :5]}")

            assert torch.allclose(y_cpu, y_cuda.cpu(), atol=1e-5, rtol=1e-5)

    @pytest.mark.skip(reason="TODO")
    @pytest.mark.unit
    def test_mixed_precision(self, dendritic_layer):
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                x = torch.randn(16, dendritic_layer.input_dim).cuda()
                y = dendritic_layer.cuda()(x)
                assert y.dtype == torch.float16

    @pytest.mark.unit
    def test_extreme_input_values(self, dendritic_layer):
        # Very large values
        x_large = torch.randn(16, dendritic_layer.input_dim) * 1e6
        y_large = dendritic_layer(x_large)
        assert not torch.isnan(y_large).any()

        # Very small values
        x_small = torch.randn(16, dendritic_layer.input_dim) * 1e-6
        y_small = dendritic_layer(x_small)
        assert not torch.isnan(y_small).any()

    @pytest.mark.unit
    def test_batch_size_edge_cases(self, dendritic_layer):
        # Batch size 1
        x = torch.randn(1, dendritic_layer.input_dim)
        y = dendritic_layer(x)
        assert y.shape == (1, dendritic_layer.output_dim)

        # Large batch size
        x = torch.randn(1024, dendritic_layer.input_dim)
        y = dendritic_layer(x)
        assert y.shape == (1024, dendritic_layer.output_dim)


# ========================
# Performance Benchmarks
# ========================
@pytest.mark.benchmark
class TestPerformance:
    @pytest.mark.parametrize("batch_size", [16, 64, 256])
    def test_forward_performance(self, benchmark, dendritic_layer, batch_size):
        x = torch.randn(batch_size, dendritic_layer.input_dim)
        benchmark(dendritic_layer, x)

    @pytest.mark.parametrize("batch_size", [16, 64, 256])
    def test_backward_performance(self, benchmark, dendritic_layer, batch_size):
        def run():
            # Create new input and output for each iteration
            x = torch.randn(batch_size, dendritic_layer.input_dim, requires_grad=True)
            y = dendritic_layer(x)
            loss = y.sum()
            loss.backward()
            dendritic_layer.zero_grad()

        benchmark(run)
