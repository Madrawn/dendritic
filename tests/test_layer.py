import torch
import torch.nn as nn
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
def dendritic_layer(layer_config):
    input_dim, output_dim, poly_rank = layer_config
    return DendriticLayer(input_dim, output_dim, poly_rank=poly_rank)

@pytest.fixture
def dendritic_stack(layer_config):
    input_dim, output_dim, poly_rank = layer_config
    return DendriticStack(input_dim, output_dim, poly_rank=poly_rank)

@pytest.fixture
def dendritic_mlp():
    return DendriticMLP(256, 1024, poly_rank=16)

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
            if 'weight' in name:
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
            layer.w1.data = torch.tensor([[1.0, 0.0]])
            layer.w2.data = torch.tensor([[0.0, 1.0]])
            layer.poly_out.data = torch.tensor([[1.0]])
            layer.linear.weight.data = torch.zeros_like(layer.linear.weight)
            layer.linear.bias.data = torch.zeros_like(layer.linear.bias)
            layer.scale.data = torch.tensor([1.0])
        
        x = torch.tensor([[2.0, 3.0]])
        y = layer(x)
        assert torch.allclose(y, torch.tensor([[6.0]]), atol=1e-4)  # 2*3 = 6

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

# ========================
# Integration & Edge Tests
# ========================
class TestIntegration:
    @pytest.mark.unit
    def test_serialization(self, dendritic_layer, layer_config):
        # Test saving and loading
        torch.save(dendritic_layer.state_dict(), 'test_layer.pth')
        input_dim, output_dim, poly_rank = layer_config
        new_layer = DendriticLayer(input_dim, output_dim, poly_rank=poly_rank)
        new_layer.load_state_dict(torch.load('test_layer.pth'))
        
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
            device = torch.device('cuda')
            print(f"\n[DEBUG] Moving layer to {device}")
            layer_cuda = dendritic_layer.to(device)
            print(f"[DEBUG] Layer device after move: {next(layer_cuda.parameters()).device}")
            
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