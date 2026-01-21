import pytest
from unittest.mock import MagicMock
from dendritic.dataset_handlers.PythonAlpacaHandler import PythonAlpacaHandler
from dendritic.enhancement import (
    NoLayersConvertedError,
    enhance_model_with_dendritic
)
from transformers.models.gpt2 import GPT2LMHeadModel


# =====================
# Model Architecture Errors
# =====================

@pytest.mark.parametrize("invalid_layer", [
    ["non_existent_layer"],
    [123],
    ["linear", "invalid.module.name"]
])
@pytest.mark.unit
def test_invalid_layer_replacement(invalid_layer):
    """Test handling of invalid layer names in enhancement"""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    with pytest.raises((TypeError, NoLayersConvertedError)) as excinfo:
        enhance_model_with_dendritic(
            model,
            target_layers=invalid_layer,
            poly_rank=8,
            freeze_linear=True
        )

# =====================
# Data Handling Errors
# =====================

@pytest.mark.unit
def test_missing_data_files(monkeypatch):
    """Test handling of missing data files in dataset handler"""
    handler = PythonAlpacaHandler(tokenizer=MagicMock(), max_length=128)
    
    # Simulate missing dataset
    monkeypatch.setattr("datasets.load_dataset", MagicMock(side_effect=FileNotFoundError("Dataset not found")))
    
    with pytest.raises(FileNotFoundError) as excinfo:
        handler.load_data()
    assert "Dataset not found" in str(excinfo.value)

@pytest.mark.unit
def test_corrupted_data_files(monkeypatch):
    """Test handling of corrupted data files"""
    handler = PythonAlpacaHandler(tokenizer=MagicMock(), max_length=128)
    
    # Simulate corrupted dataset
    monkeypatch.setattr("datasets.load_dataset", MagicMock(side_effect=RuntimeError("Corrupted data")))
    
    with pytest.raises(RuntimeError) as excinfo:
        handler.load_data()
    assert "Corrupted data" in str(excinfo.value)

# =====================
# Configuration Errors
# =====================

@pytest.mark.parametrize("invalid_config", [
    {"poly_rank": -1},
    {"dendritic_cls": "invalid_class"},
    {"dendritic_kwargs": "not_a_dict"}
])
@pytest.mark.unit
def test_invalid_enhancement_parameters(invalid_config):
    """Test invalid parameters for model enhancement"""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    with pytest.raises((NoLayersConvertedError, ValueError)) as excinfo:
        enhance_model_with_dendritic(model, **invalid_config)


# =====================
# Network Failure Handling
# =====================

@pytest.mark.unit
def test_network_failure_during_download(monkeypatch):
    """Test network failures during dataset download"""
    handler = PythonAlpacaHandler(tokenizer=MagicMock(), max_length=128)
    
    # Simulate network error
    monkeypatch.setattr("datasets.load_dataset", MagicMock(side_effect=ConnectionError("Network failure")))
    
    with pytest.raises(ConnectionError) as excinfo:
        handler.load_data()
    assert "Network failure" in str(excinfo.value)




