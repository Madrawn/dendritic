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


# # New tests for dataset handler refactoring
# @pytest.mark.unit
# def test_factory_pattern(mock_tokenizer):
#     """Test handler factory registration and retrieval"""
#     # Test getting existing handlers
#     wikitext_handler = get_handler('wikitext', mock_tokenizer)
#     assert isinstance(wikitext_handler, WikiTextHandler)
    
#     openwebmath_handler = get_handler('openwebmath', mock_tokenizer)
#     assert isinstance(openwebmath_handler, OpenWebMathHandler)
    
#     # Test invalid handler
#     with pytest.raises(KeyError, match="not registered"):
#         get_handler('nonexistent', mock_tokenizer)


# @pytest.mark.unit
# def test_openwebmath_handler_interface(mock_tokenizer):
#     """Test OpenWebMathHandler implements required interface"""
#     handler = OpenWebMathHandler(mock_tokenizer)
    
#     assert hasattr(handler, 'load_default_data')
#     assert hasattr(handler, 'tokenize_function')
#     assert hasattr(handler, 'prepare_data')
#     assert hasattr(handler, 'prepare_pretraining_data')
    
#     # Verify it inherits from TextCorpusHandler
#     assert isinstance(handler, TextCorpusHandler)


# @pytest.mark.unit
# def test_wikitext_handler_streaming_support(mock_tokenizer, mocker):
#     """Test WikiTextHandler streaming mode"""
#     handler = WikiTextHandler(mock_tokenizer)
    
#     # Mock load_dataset to return IterableDataset for streaming
#     mock_train = mocker.Mock(spec=IterableDataset)
#     mock_val = mocker.Mock(spec=IterableDataset)
#     mock_train.column_names = ['text']
#     mock_val.column_names = ['text']
    
#     mocker.patch('datasets.load_dataset', side_effect=[
#         mock_train,  # First call for train split
#         mock_val     # Second call for validation split
#     ])
    
#     # Test streaming mode
#     data = handler.load_default_data(streaming=True)
#     assert 'train' in data
#     assert 'test' in data
#     assert isinstance(data['train'], IterableDataset)
#     assert isinstance(data['test'], IterableDataset)


# @pytest.mark.unit
# def test_text_corpus_handler_pretraining_data(mock_tokenizer, mocker):
#     """Test TextCorpusHandler.prepare_pretraining_data method"""
#     # Create a mock handler
#     class MockTextCorpusHandler(TextCorpusHandler):
#         def __init__(self, tokenizer, max_length=256):
#             super().__init__(tokenizer, max_length, dataset_name="test", text_column="text")
        
#         def load_default_data(self, split="train", test_size=0.1, seed=42, streaming=False, **kwargs):
#             # Return mock datasets
#             train_ds = Dataset.from_dict({'text': ['sample text 1', 'sample text 2']})
#             test_ds = Dataset.from_dict({'text': ['test text']})
#             return {'train': train_ds, 'test': test_ds}
    
#     handler = MockTextCorpusHandler(mock_tokenizer)
    
#     # Mock config
#     class MockConfig:
#         training_steps = 10
#         batch_size = 2
#         max_seq_len = 256
    
#     config = MockConfig()
    
#     # Test non-streaming preparation
#     prepared = handler.prepare_pretraining_data(config, num_workers=0)
    
#     assert 'train' in prepared
#     assert 'eval' in prepared
#     assert hasattr(prepared['train'], '__iter__')  # Should be a DataLoader
#     assert hasattr(prepared['eval'], '__iter__')


# @pytest.mark.unit
# def test_handler_param_passing(mock_tokenizer, mocker):
#     """Test that handler parameters are passed correctly through factory"""
#     # Mock load_dataset to track calls
#     mock_load = mocker.patch('datasets.load_dataset')
#     mock_dataset = Dataset.from_dict({'text': ['test']})
#     mock_load.return_value = mock_dataset
    
#     # Get handler with custom parameters
#     handler = get_handler('wikitext', mock_tokenizer, max_length=512)
    
#     # Call load_default_data with additional params
#     handler.load_default_data(test_size=0.2, seed=123, custom_param='value')
    
#     # Verify load_dataset was called with correct parameters
#     mock_load.assert_called()
#     call_kwargs = mock_load.call_args[1]
#     assert 'custom_param' not in call_kwargs  # Should be filtered out
#     # The actual dataset loading happens with split parameter


# @pytest.mark.unit
# def test_backward_compatibility(mock_tokenizer):
#     """Test backward compatibility - WikiTextHandler should still work"""
#     # Test that WikiTextHandler works with original interface
#     handler = WikiTextHandler(mock_tokenizer)
#     assert hasattr(handler, 'prepare_data')
#     assert hasattr(handler, 'tokenize_function')
#     assert hasattr(handler, 'load_default_data')
    
#     # Test that factory returns correct handler
#     handler = get_handler('wikitext', mock_tokenizer)
#     assert isinstance(handler, WikiTextHandler)


# @pytest.mark.integration
# def test_end_to_end_handler_usage(mock_tokenizer, mocker):
#     """Integration test for complete handler workflow"""
#     # Mock the actual dataset loading to avoid network calls
#     mock_dataset = Dataset.from_dict({
#         'text': ['This is a sample document for testing.', 'Another document here.']
#     })
#     mocker.patch('datasets.load_dataset', return_value=mock_dataset)
    
#     # Test OpenWebMath handler
#     handler = OpenWebMathHandler(mock_tokenizer, max_length=128)
    
#     # Load data
#     data = handler.load_default_data(test_size=0.3)
#     assert 'train' in data
#     assert 'test' in data
    
#     # Prepare data
#     prepared = handler.prepare_data()
#     assert 'train' in prepared
#     assert 'eval' in prepared
    
#     # Mock config for pretraining
#     class MockConfig:
#         training_steps = 5
#         batch_size = 1
#         max_seq_len = 128
    
#     config = MockConfig()
    
#     # Test pretraining preparation
#     pretraining_data = handler.prepare_pretraining_data(config, num_workers=0)
#     assert 'train' in pretraining_data
#     assert 'eval' in pretraining_data


# @pytest.mark.edge
# def test_streaming_with_empty_dataset(mock_tokenizer, mocker):
#     """Edge case test: streaming with potentially empty dataset"""
#     handler = WikiTextHandler(mock_tokenizer)
    
#     # Mock empty IterableDataset
#     mock_empty_ds = mocker.Mock(spec=IterableDataset)
#     mock_empty_ds.column_names = ['text']
#     mock_empty_ds.__iter__ = mocker.Mock(return_value=iter([]))  # Empty iterator
    
#     mocker.patch('datasets.load_dataset', side_effect=[
#         mock_empty_ds,  # Train
#         mock_empty_ds   # Validation
#     ])
    
#     # Should not crash with empty streaming dataset
#     data = handler.load_default_data(streaming=True)
#     assert data['train'] is not None
#     assert data['test'] is not None
