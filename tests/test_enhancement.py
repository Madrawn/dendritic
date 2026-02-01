import pytest
import torch
import torch.nn as nn
from dendritic.enhancement import (
    NoLayersConvertedError,
    apply_dendritic_state,
    enhance_model_with_dendritic,
    verify_identity_initialization,
    get_polynomial_stats,
    extract_dendritic_state,
)
from dendritic.layers.DendriticLayer import DendriticLayer
from dendritic.layers.DendriticStack import DendriticStack
from unittest.mock import patch, MagicMock

# =====================
# Mock Models & Fixtures
# =====================


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MockConv1D(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if True else None

    def forward(self, x):
        return x @ self.weight.t() + self.bias


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(64, 8)
        self.mlp = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)
        mlp_out = self.mlp(x)
        x = x + mlp_out
        return self.ln2(x)


@pytest.fixture
def simple_mlp():
    return SimpleMLP()


@pytest.fixture
def transformer_block():
    return TransformerBlock()


@pytest.fixture
def mock_conv1d():
    model = nn.Sequential(MockConv1D(64, 128), nn.ReLU(), MockConv1D(128, 64))
    return model


# =====================
# Core Enhancement Tests
# =====================


@pytest.mark.unit
def test_basic_enhancement(simple_mlp):
    """Test basic model enhancement with default parameters"""
    original_params = sum(p.numel() for p in simple_mlp.parameters())
    enhanced = enhance_model_with_dendritic(simple_mlp)

    # Verify parameter increase
    enhanced_params = sum(p.numel() for p in enhanced.parameters())
    assert enhanced_params > original_params

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in enhanced.parameters() if p.requires_grad)
    assert trainable_params > 0
    assert (
        trainable_params < enhanced_params
    )  # Only dendritic params should be trainable


@pytest.mark.unit
def test_dendritic_layer_types(simple_mlp):
    """Test different dendritic layer types (individual, stack, mlp)"""
    # Test DendriticLayer
    enhanced_layer = enhance_model_with_dendritic(
        simple_mlp, dendritic_cls=DendriticLayer
    )
    assert any(isinstance(m, DendriticLayer) for m in enhanced_layer.modules())

    # Test DendriticStack
    enhanced_stack = enhance_model_with_dendritic(
        simple_mlp, dendritic_cls=DendriticStack
    )
    assert any(isinstance(m, DendriticStack) for m in enhanced_stack.modules())


@pytest.mark.unit
def test_layer_placement_strategies(simple_mlp):
    """Test layer placement strategies (interleave, replace_mlp, etc.)"""
    # Test selective layer enhancement
    enhanced_selective = enhance_model_with_dendritic(simple_mlp, target_layers=["fc1"])
    named_modules = [*enhanced_selective.named_modules()]
    fc1_replaced = any(
        isinstance(m, (DendriticLayer, DendriticStack))
        for name, m in named_modules
        if "fc1" in name
    )
    fc2_not_replaced = all(
        not isinstance(m, (DendriticLayer, DendriticStack))
        for name, m in named_modules
        if "fc2" in name
    )
    assert fc1_replaced
    assert fc2_not_replaced


@pytest.mark.unit
def test_enhancement_ratios(simple_mlp):
    """Test enhancement ratios and scaling factors"""
    # Test auto poly_rank
    enhanced_auto = enhance_model_with_dendritic(
        simple_mlp, poly_rank="auto", target_layers=["fc3"]
    )
    dendritic_layer = next(
        m
        for m in enhanced_auto.modules()
        if isinstance(m, (DendriticLayer, DendriticStack))
    )
    assert dendritic_layer.poly_rank == max(4, 256 // 64)


@pytest.mark.unit
def test_enhancement_fixed(simple_mlp):
    # Test fixed poly_rank
    enhanced_fixed = enhance_model_with_dendritic(simple_mlp, poly_rank=32)
    dendritic_layer = next(
        m
        for m in enhanced_fixed.modules()
        if isinstance(m, (DendriticLayer, DendriticStack))
    )
    assert dendritic_layer.poly_rank == 32


@pytest.mark.unit
@pytest.mark.unit
def test_selective_layer_enhancement(simple_mlp):
    """Test selective layer enhancement"""
    enhanced = enhance_model_with_dendritic(simple_mlp, target_layers=["fc1", "fc3"])

    replaced_layers = [
        name
        for name, m in enhanced.named_modules()
        if isinstance(m, (DendriticLayer, DendriticStack))
    ]

    assert any("fc1" in name for name in replaced_layers)
    assert any("fc3" in name for name in replaced_layers)
    assert not any("fc2" in name for name in replaced_layers)


# =====================
# Parameter Isolation Tests
# =====================


@pytest.mark.unit
def test_parameter_isolation(simple_mlp):
    """Test dendritic parameter identification and isolation"""
    enhanced = enhance_model_with_dendritic(simple_mlp, freeze_linear=True)

    # Verify original weights are frozen
    for name, param in enhanced.named_parameters():
        if "linear" in name or "base_linear" in name:
            assert not param.requires_grad
        elif "w1" in name or "w2" in name or "scale" in name or "diag" in name:
            assert param.requires_grad


@pytest.mark.unit
def test_gradient_flow(simple_mlp):
    """Test gradient flow preservation"""
    enhanced = enhance_model_with_dendritic(simple_mlp)
    input_tensor = torch.randn(2, 64)
    output = enhanced(input_tensor)
    loss = output.sum()
    loss.backward()

    # Verify gradients exist for dendritic params
    for name, param in enhanced.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


# =====================
# Error Handling Tests
# =====================


@pytest.mark.unit
def test_invalid_model_architecture():
    """Test invalid model architectures"""
    invalid_model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU())
    with pytest.raises(NoLayersConvertedError):
        enhance_model_with_dendritic(invalid_model)


@pytest.mark.unit
def test_unsupported_layer_type():
    """Test unsupported layer types"""
    model = nn.Sequential(nn.Linear(64, 128), nn.Conv2d(128, 256, 3))
    with pytest.raises(NoLayersConvertedError):
        enhance_model_with_dendritic(model, target_layers=["1"])


@pytest.mark.unit
def test_invalid_enhancement_ratios(simple_mlp):
    """Test invalid enhancement ratios"""
    with pytest.raises(ValueError):
        enhance_model_with_dendritic(simple_mlp, poly_rank=-5)


# =====================
# State Management Tests
# =====================


@pytest.mark.unit
def test_model_state_preservation(simple_mlp):
    """Test model state preservation during enhancement"""
    original_state = simple_mlp.state_dict()
    enhanced = enhance_model_with_dendritic(simple_mlp)

    # Verify original weights unchanged
    for name, param in enhanced.named_parameters():
        if "linear" in name and name in original_state:
            assert torch.allclose(param.data, original_state[name])


@pytest.mark.unit
def test_layer_config_serialization(simple_mlp):
    """Test layer configuration serialization"""
    enhanced = enhance_model_with_dendritic(
        simple_mlp, poly_rank=16, init_scale=1e-4, dendritic_kwargs={"dropout": 0.2}
    )

    # Verify config attributes
    dendritic_layer = next(
        m for m in enhanced.modules() if isinstance(m, (DendriticLayer, DendriticStack))
    )
    assert dendritic_layer.poly_rank == 16
    if hasattr(dendritic_layer, "dropout"):
        dropout_attr = dendritic_layer.dropout
        # Handle both nn.Dropout and Tensor cases
        if isinstance(dropout_attr, nn.Dropout):
            assert dropout_attr.p == 0.2
        elif isinstance(dropout_attr, torch.Tensor):
            assert torch.allclose(dropout_attr, torch.tensor(0.2))


@pytest.mark.unit
def test_enhancement_reversibility(simple_mlp):
    """Test enhancement reversibility where applicable"""
    # Not directly reversible, but we can verify identity initialization
    enhanced = enhance_model_with_dendritic(simple_mlp)
    input_tensor = torch.randn(2, 64)

    # Verify outputs match before training
    diff = verify_identity_initialization(simple_mlp, enhanced, input_tensor)
    assert diff < 1e-6


@pytest.mark.unit
def test_extract_and_load_dendritic_state(simple_mlp):
    """Test dendritic state extraction and loading"""
    enhanced = enhance_model_with_dendritic(simple_mlp)
    state = extract_dendritic_state(enhanced)

    # Verify state contains metadata
    assert "_metadata" in state

    # Create new enhanced model and load state
    new_enhanced = apply_dendritic_state(SimpleMLP(), state)

    # Verify parameters match
    for (n1, p1), (n2, p2) in zip(
        enhanced.named_parameters(), new_enhanced.named_parameters()
    ):
        if p1.requires_grad:
            assert torch.allclose(p1, p2)


# =====================
# Helper Function Tests
# =====================


@pytest.mark.unit
def test_verify_identity_initialization(simple_mlp):
    """Test identity initialization verification"""
    enhanced = enhance_model_with_dendritic(simple_mlp)
    input_tensor = torch.randn(2, 64)
    diff = verify_identity_initialization(simple_mlp, enhanced, input_tensor)
    assert diff < 1e-6


@pytest.mark.unit
def test_get_polynomial_stats(simple_mlp):
    """Test polynomial stats collection"""
    enhanced = enhance_model_with_dendritic(simple_mlp)
    stats = get_polynomial_stats(enhanced)

    assert len(stats) > 0
    for layer_stats in stats.values():
        assert "scale" in layer_stats
        assert "poly_rank" in layer_stats


@pytest.mark.unit
def test_create_dendritic_state(simple_mlp):
    """Test combined enhancement and state loading"""
    state = extract_dendritic_state(enhance_model_with_dendritic(simple_mlp))
    new_enhanced = apply_dendritic_state(SimpleMLP(), state_payload=state)

    # Verify it's enhanced
    assert any(
        isinstance(m, (DendriticLayer, DendriticStack)) for m in new_enhanced.modules()
    )
