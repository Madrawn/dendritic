import pytest
import torch
import numpy as np
from dendritic.dataset_handlers import PythonAlpacaHandler, BaseDatasetHandler
from dendritic.enhancement import enhance_model_with_dendritic
from layers.DendriticLayer import DendriticLayer
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import Dataset
from typing import Dict, Any
import time
import os
import json
import csv
import tempfile

# Mock tokenizer for testing
class MockTokenizer:
    def __init__(self):
        self.eos_token = '<EOS>'
        self.pad_token = '<PAD>'
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab_size = 10000

    def __call__(self, texts, truncation=False, padding=False, max_length=None, return_tensors=None):
        # Simple tokenization: split by space
        # Return as a dictionary with 'input_ids' as list of lists
        return {'input_ids': [[len(word) for word in text.split()] for text in texts]}

@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()

@pytest.fixture
def alpaca_handler(mock_tokenizer):
    return PythonAlpacaHandler(mock_tokenizer, max_length=256)

@pytest.fixture
def sample_dataset_dict():
    return {
        'train': Dataset.from_dict({
            'prompt': [
                "### Instruction:\nWrite a function to add two numbers\n### Input:\nNone\n### Output:",
                "### Instruction:\nWrite a function to subtract numbers\n### Input:\nNone\n### Output:"
            ],
            'output': [
                "def add(a, b): return a + b",
                "def subtract(a, b): return a - b"
            ]
        }),
        'test': Dataset.from_dict({
            'prompt': [
                "### Instruction:\nWrite a function to multiply two numbers\n### Input:\nNone\n### Output:"
            ],
            'output': [
                "def multiply(a, b): return a * b"
            ]
        })
    }

# =====================
# Data Pipeline Integration Tests
# =====================

@pytest.mark.unit
def test_complete_data_flow(alpaca_handler, sample_dataset_dict):
    """Test complete data flow: raw data -> dataset handler -> preprocessing -> model input"""
    # Mock the load_data method to return our sample dataset
    alpaca_handler.load_data = lambda **kwargs: sample_dataset_dict
    
    # Prepare the data
    prepared_data = alpaca_handler.prepare_data()
    
    # Check that we have train and eval datasets
    assert 'train' in prepared_data
    assert 'eval' in prepared_data
    
    # Check the format of the train dataset
    train_dataset = prepared_data['train']
    assert isinstance(train_dataset, Dataset)
    # Check that the expected keys are present
    example = train_dataset[0]
    assert 'input_ids' in example
    assert 'attention_mask' in example
    assert 'labels' in example
    
    # Check the shape of input_ids and labels
    assert len(example['input_ids']) == 256
    assert len(example['labels']) == 256
    
    # Check that the labels have the correct masking: -100 for non-output tokens
    assert any(label == -100 for label in example['labels'])
    assert any(label != -100 for label in example['labels'])

@pytest.mark.unit
def test_batch_collation(alpaca_handler, sample_dataset_dict):
    """Test batch collation and padding strategies"""
    alpaca_handler.load_data = lambda **kwargs: sample_dataset_dict
    prepared_data = alpaca_handler.prepare_data()
    train_dataset = prepared_data['train']
    
    # Create a DataLoader to test batching
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=lambda x: x)
    
    batch = next(iter(dataloader))
    assert len(batch) == 2
    
    # Check that all items in batch have same length
    input_ids_lengths = [len(item['input_ids']) for item in batch]
    assert len(set(input_ids_lengths)) == 1
    
    # Check attention masks are correct
    for item in batch:
        assert sum(item['attention_mask']) == sum(1 for id in item['input_ids'] if id != alpaca_handler.tokenizer.pad_token_id)

# =====================
# Data Consistency Tests
# =====================

@pytest.mark.unit
def test_data_integrity(alpaca_handler, sample_dataset_dict):
    """Test data integrity throughout the pipeline"""
    alpaca_handler.load_data = lambda **kwargs: sample_dataset_dict
    prepared_data = alpaca_handler.prepare_data()
    train_dataset = prepared_data['train']
    
    # Get two examples
    example1 = train_dataset[0]
    example2 = train_dataset[1]
    
    # Check that the tokenization is consistent
    # (same prompt should produce same tokenization, different prompts should be different)
    if example1['input_ids'][:10].equal(example2['input_ids'][:10]):
        # If the first 10 tokens are the same, the prompts must be similar
        prompt1 = sample_dataset_dict['train']['prompt'][0]
        prompt2 = sample_dataset_dict['train']['prompt'][1]
        assert prompt1[:20] == prompt2[:20]
    else:
        assert example1['input_ids'][:10].equal(example2['input_ids'][:10]) is False

@pytest.mark.unit
def test_tokenization_consistency(mock_tokenizer):
    """Test tokenization consistency across different handlers"""
    # Create two handlers with the same tokenizer instance
    handler1 = PythonAlpacaHandler(mock_tokenizer, max_length=256)
    handler2 = PythonAlpacaHandler(mock_tokenizer, max_length=256)
    
    sample = {"prompt": ["Test prompt"], "output": ["Test output"]}
    
    tokenized1 = handler1.tokenize_function(sample)
    tokenized2 = handler2.tokenize_function(sample)
    
    # Should produce identical results with same tokenizer
    assert tokenized1['input_ids'] == tokenized2['input_ids']
    assert tokenized1['attention_mask'] == tokenized2['attention_mask']
    assert tokenized1['labels'] == tokenized2['labels']

# =====================
# Data Validation Tests
# =====================

@pytest.mark.unit
def test_input_validation(alpaca_handler, sample_dataset_dict):
    """Test input validation at each pipeline stage"""
    # Test with malformed data
    malformed_data = sample_dataset_dict.copy()
    malformed_data['train'] = malformed_data['train'].add_column('invalid_column', [1, 2])
    
    alpaca_handler.load_data = lambda **kwargs: malformed_data
    prepared_data = alpaca_handler.prepare_data()
    
    # Should still work, extra columns should be removed
    assert 'invalid_column' not in prepared_data['train'].column_names

@pytest.mark.unit
def test_data_type_validation(alpaca_handler):
    """Test data type consistency"""
    invalid_data = {
        'prompt': [123],  # Integer instead of string
        'output': [456]    # Integer instead of string
    }
    
    with pytest.raises(TypeError):
        alpaca_handler.tokenize_function(invalid_data)

@pytest.mark.unit
def test_shape_validation(alpaca_handler):
    """Test shape and dimension validation"""
    # Create data with mismatched lengths
    invalid_data = {
        'prompt': ["Short prompt"],
        'output': ["Longer output that exceeds max length" * 100]
    }
    
    tokenized = alpaca_handler.tokenize_function(invalid_data)
    assert len(tokenized['input_ids'][0]) == 256
    assert len(tokenized['labels'][0]) == 256

# =====================
# Pipeline Error Handling
# =====================

@pytest.mark.unit
def test_malformed_data_handling(alpaca_handler):
    """Test malformed data detection"""
    malformed_data = {
        'prompt': [None],
        'output': ["Valid output"]
    }
    
    with pytest.raises(TypeError):
        alpaca_handler.tokenize_function(malformed_data)

@pytest.mark.unit
def test_data_corruption_handling(alpaca_handler):
    """Test data corruption handling"""
    # Create corrupted sample (invalid UTF-8)
    corrupted_data = {
        'prompt': [b"Invalid \x80\x99 string".decode('utf-8', errors='replace')],
        'output': ["Valid output"]
    }
    
    # Should handle gracefully
    tokenized = alpaca_handler.tokenize_function(corrupted_data)
    assert tokenized['input_ids']

@pytest.mark.unit
def test_missing_data_handling(alpaca_handler):
    """Test missing data handling"""
    missing_data = {
        'prompt': ["Valid prompt"],
        'output': [None]
    }
    
    with pytest.raises(TypeError):
        alpaca_handler.tokenize_function(missing_data)

# =====================
# Performance and Scalability
# =====================

@pytest.mark.unit
def test_pipeline_throughput(alpaca_handler, sample_dataset_dict):
    """Test pipeline throughput"""
    alpaca_handler.load_data = lambda **kwargs: sample_dataset_dict
    
    start_time = time.time()
    prepared_data = alpaca_handler.prepare_data()
    end_time = time.time()
    
    # Should process small dataset quickly
    assert end_time - start_time < 1.0

@pytest.mark.unit
def test_large_dataset_handling(alpaca_handler):
    """Test large dataset handling"""
    # Create a large dataset
    large_data = {
        'prompt': ["Sample prompt"] * 1000,
        'output': ["Sample output"] * 1000
    }
    large_dataset = Dataset.from_dict(large_data)
    sample_dataset_dict = {'train': large_dataset, 'test': large_dataset}
    
    alpaca_handler.load_data = lambda **kwargs: sample_dataset_dict
    
    start_time = time.time()
    prepared_data = alpaca_handler.prepare_data()
    end_time = time.time()
    
    # Should handle 1000 samples reasonably
    assert end_time - start_time < 5.0
    assert len(prepared_data['train']) == 1000

# =====================
# Multi-Format Support
# =====================

@pytest.mark.unit
def test_json_format_support(alpaca_handler):
    """Test JSON format support"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([
            {"prompt": "JSON prompt 1", "output": "JSON output 1"},
            {"prompt": "JSON prompt 2", "output": "JSON output 2"}
        ], f)
        f.close()
        
        # Should be able to load JSON
        dataset = alpaca_handler.load_data(data_files=f.name)
        assert len(dataset['train']) == 2
        os.unlink(f.name)

@pytest.mark.unit
def test_csv_format_support(alpaca_handler):
    """Test CSV format support"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=['prompt', 'output'])
        writer.writeheader()
        writer.writerow({'prompt': 'CSV prompt 1', 'output': 'CSV output 1'})
        writer.writerow({'prompt': 'CSV prompt 2', 'output': 'CSV output 2'})
        f.close()
        
        # Should be able to load CSV
        dataset = alpaca_handler.load_data(data_files=f.name)
        assert len(dataset['train']) == 2
        os.unlink(f.name)

@pytest.mark.unit
def test_text_file_support(alpaca_handler):
    """Test text file support"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Text file line 1\n")
        f.write("Text file line 2\n")
        f.close()
        
        # Should be able to load text files
        dataset = alpaca_handler.load_data(data_files=f.name)
        assert len(dataset['train']) == 2
        os.unlink(f.name)


# =====================
# Test Custom Data Formats
# =====================

class CustomDatasetHandler(BaseDatasetHandler):
    """Custom handler for testing"""
    def load_default_data(self, **kwargs) -> Dict[str, Any]:
        return {
            'train': Dataset.from_dict({'text': ['Sample text 1', 'Sample text 2']}),
            'test': Dataset.from_dict({'text': ['Test text']})
        }
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        # Convert tokenizer output to dictionary for consistent access
        tokenized = dict(self.tokenizer(examples['text'], truncation=True, max_length=self.max_length))
        input_ids = tokenized['input_ids']
        
        # Ensure input_ids is a list before processing
        assert isinstance(input_ids, list), "input_ids must be a list"
        
        # Create attention_mask and labels
        attention_mask = [[1] * len(ids) for ids in input_ids]
        labels = [ids.copy() for ids in input_ids]  # copy each id list
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

@pytest.mark.unit
def test_custom_data_handler(mock_tokenizer):
    """Test support for custom data formats"""
    handler = CustomDatasetHandler(mock_tokenizer, max_length=128)
    prepared_data = handler.prepare_data()
    
    assert 'train' in prepared_data
    assert 'eval' in prepared_data
    assert len(prepared_data['train']) == 2