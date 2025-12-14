import pytest
from dendritic.dataset_handlers.BaseDatasetHandler import BaseDatasetHandler
from dendritic.dataset_handlers.PythonAlpacaHandler import PythonAlpacaHandler
from dendritic.dataset_handlers.dataset_handlers import (
    GitHubCodeHandler,
    TinyStoriesHandler,
    WikiTextHandler
)
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import Dataset
from typing import Dict, Any
import torch

from test_data_pipelines import MockTokenizer



@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()

@pytest.fixture
def alpaca_handler(mock_tokenizer):
    return PythonAlpacaHandler(mock_tokenizer, max_length=256)

@pytest.fixture
def alpaca_sample_data():
    return {
        'prompt': [
            "### Instruction:\nWrite a function to add two numbers\n### Input:\nNone\n### Output:",
            "### Instruction:\nWrite a function to subtract numbers\n### Input:\nNone\n### Output:"
        ],
        'output': [
            "def add(a, b): return a + b",
            "def subtract(a, b): return a - b"
        ]
    }

# Test PythonAlpacaHandler functionality
@pytest.mark.unit
def test_alpaca_handler_load_data(alpaca_handler, mocker):
    """Test dataset loading and splitting functionality"""
    mock_load = mocker.patch('datasets.load_dataset')
    mock_load.return_value = Dataset.from_dict({
        'prompt': ['test1', 'test2'],
        'output': ['out1', 'out2']
    })
    
    data = alpaca_handler.load_data(split='train', test_size=0.2)
    assert 'train' in data
    assert 'test' in data
    assert len(data['train']) == 1
    # Accept either ['prompt', 'output'] or ['output', 'prompt'] depending on Dataset implementation
    assert set(data['train'].column_names) == {'prompt', 'output'}

@pytest.mark.unit
def test_alpaca_tokenize_normal(alpaca_handler, alpaca_sample_data):
    """Test tokenization with valid data"""
    tokenized = alpaca_handler.tokenize_function(alpaca_sample_data)
    
    assert 'input_ids' in tokenized
    assert 'attention_mask' in tokenized
    assert 'labels' in tokenized
    assert len(tokenized['input_ids']) == 2
    assert len(tokenized['input_ids'][0]) == 256

@pytest.mark.unit
def test_alpaca_tokenize_malformed(alpaca_handler):
    """Test tokenization error handling for malformed data"""
    malformed_data = {
        'prompt': ["Prompt without output marker"],
        'output': [None]  # Invalid type
    }
    
    with pytest.raises(TypeError):
        alpaca_handler.tokenize_function(malformed_data)

@pytest.mark.unit
def test_alpaca_prepare_data(alpaca_handler, mocker):
    """Test end-to-end data preparation"""
    mock_load = mocker.patch('datasets.load_dataset')
    mock_data = Dataset.from_dict({
        'prompt': ['test1', 'test2'],
        'output': ['out1', 'out2']
    })
    mock_load.return_value = mock_data
    print(f"DEBUG: Mock return type: {type(mock_data)}")  # Add debug log
    
    prepared = alpaca_handler.prepare_data()
    assert 'train' in prepared
    assert 'eval' in prepared
    assert prepared['train'].format['type'] == 'torch'


@pytest.mark.parametrize("handler_class", [
    # GitHubCodeHandler,
    # TinyStoriesHandler,
    # WikiTextHandler,
    PythonAlpacaHandler
])
@pytest.mark.unit
def test_handler_interface_compliance(handler_class, mock_tokenizer):
    """Test all handlers implement required interface"""
    handler = handler_class(mock_tokenizer)
    
    assert hasattr(handler, 'load_data')
    assert hasattr(handler, 'tokenize_function')
    assert hasattr(handler, 'prepare_data')
    
    # Verify method signatures
    assert callable(handler.load_data)
    assert callable(handler.tokenize_function)
    assert callable(handler.prepare_data)

@pytest.mark.unit
def test_base_class_prepare_data(mocker, mock_tokenizer):
    """Test base class prepare_data method structure"""
    class ConcreteHandler(BaseDatasetHandler):
        def load_default_data(self, **kwargs):
            return {'train': Dataset.from_dict({'text': ['sample']}),
                    'test': Dataset.from_dict({'text': ['test']})}
        
        def tokenize_function(self, examples):
            return {
                'input_ids': [[1,2,3]],
                'attention_mask': [[1,1,1]],
                'labels': [[1,2,3]]
            }
    
    handler = ConcreteHandler(mock_tokenizer)
    prepared = handler.prepare_data()
    
    assert 'train' in prepared
    assert 'eval' in prepared
    assert prepared['train'].format['type'] == 'torch'

@pytest.mark.unit
def test_module_exports():
    """Test that all handlers are properly exported"""
    from dendritic.dataset_handlers import dataset_handlers
    expected_exports = [
        'BaseDatasetHandler',
        'PythonAlpacaHandler',
        'GitHubCodeHandler',
        'TinyStoriesHandler',
        'WikiTextHandler'
    ]
    for export in expected_exports:
        assert hasattr(dataset_handlers, export), f"Missing export: {export}"