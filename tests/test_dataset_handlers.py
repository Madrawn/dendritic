import pytest
import dendritic
from dendritic.dataset_handlers.BaseDatasetHandler import BaseDatasetHandler
from dendritic.dataset_handlers.PythonAlpacaHandler import PythonAlpacaHandler
from dendritic.dataset_handlers.dataset_handlers import (
    GitHubCodeHandler,
    TinyStoriesHandler,
    WikiTextHandler,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import Dataset, IterableDataset
import torch

import dendritic.experiments
import dendritic.experiments.utils
import dendritic.experiments.utils.PretrainingConfig
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
        "prompt": [
            "### Instruction:\nWrite a function to add two numbers\n### Input:\nNone\n### Output:",
            "### Instruction:\nWrite a function to subtract numbers\n### Input:\nNone\n### Output:",
        ],
        "output": ["def add(a, b): return a + b", "def subtract(a, b): return a - b"],
    }


@pytest.mark.unit
def test_wikitext_handler_load_data(wikitext_handler, mocker):
    """Test WikiText dataset loading and splitting functionality"""
    mock_load = mocker.patch("dendritic.dataset_handlers.TextCorpusHandler.load_dataset")
    mock_load.return_value = Dataset.from_dict({"text": ["sample1", "sample2"] * 100})
    data = wikitext_handler.load_data(split="train", test_size=0.5, max_samples=200)
    assert "train" in data
    assert "test" in data
    assert len(data["train"]) == 100
    assert set(data["train"].column_names) == {"text"}


@pytest.mark.unit
def test_wikitext_tokenize_normal(wikitext_handler, wikitext_sample_data):
    """Test WikiText tokenization with valid data"""
    tokenized = wikitext_handler.tokenize_function(wikitext_sample_data)
    assert "input_ids" in tokenized
    assert "labels" in tokenized
    assert len(tokenized["input_ids"]) == 2
    assert len(tokenized["input_ids"][0]) == 7


@pytest.mark.unit
def test_wikitext_tokenize_malformed(wikitext_handler):
    """Test WikiText tokenization error handling for malformed data"""
    malformed_data = {"text": [None]}
    with pytest.raises(Exception):
        wikitext_handler.tokenize_function(malformed_data)


@pytest.mark.timeout(120)
@pytest.mark.unit
def test_wikitext_prepare_data(wikitext_handler: WikiTextHandler, mocker):
    """Test WikiText end-to-end data preparation"""
    mock_load = mocker.patch("dendritic.dataset_handlers.TextCorpusHandler.load_dataset")
    # Use longer samples to ensure they don't get filtered out or result in empty blocks
    long_text = "sample text " * 1000
    mock_data = IterableDataset.from_generator(
        lambda: (x for x in ([{"text": long_text}] * 5000))
    )
    mock_load.return_value = mock_data
    cfg = dendritic.experiments.utils.PretrainingConfig.PretrainingConfig(
        training_steps=100, batch_size=2, max_seq_len=256, eval_split_ratio=0.5, grouped_by_length=True, group_separator="EOS_token"
    )
    prepared = wikitext_handler.prepare_pretraining_dataloaders(config=cfg)
    print(f"DEBUG prepared keys: {list(prepared.keys())}")
    print(
        f"DEBUG train dataset length: {len(prepared['train'].dataset) if hasattr(prepared['train'], 'dataset') else 'no dataset'}"
    )
    print(
        f"DEBUG eval dataset length: {len(prepared['eval'].dataset) if hasattr(prepared['eval'], 'dataset') else 'no dataset'}"
    )
    assert "train" in prepared
    assert "eval" in prepared
    train_length = sum(1 for _ in prepared["train"])
    eval_length = sum(1 for _ in prepared["eval"])
    print(f"DEBUG train_length (batches): {train_length}, eval_length: {eval_length}")
    # with training_steps=100, batch_size=2, eval_split_ratio=0.5 we expect samples
    assert train_length >= int(100)
    assert eval_length >= int(train_length * 0.5)
    # For DataLoader, we check format on the underlying dataset
    assert prepared["train"].dataset.format["type"] == "torch"


@pytest.fixture
def wikitext_handler(mock_tokenizer):
    return WikiTextHandler(mock_tokenizer, max_length=256)


@pytest.fixture
def wikitext_sample_data():
    return {
        "text": [
            "This is a sample sentence from WikiText.",
            "Another WikiText example for testing.",
        ]
    }


# Test PythonAlpacaHandler functionality


@pytest.mark.unit
def test_alpaca_handler_load_data(alpaca_handler, mocker):
    """Test dataset loading and splitting functionality"""
    mock_load = mocker.patch("datasets.load_dataset")
    mock_load.return_value = Dataset.from_dict(
        {"prompt": ["test1", "test2"] * 100, "output": ["out1", "out2"] * 100}
    )

    data = alpaca_handler.load_data(split="train", test_size=0.5)
    assert "train" in data
    assert "test" in data
    assert len(data["train"]) == 100
    # Accept either ['prompt', 'output'] or ['output', 'prompt'] depending on Dataset implementation
    assert set(data["train"].column_names) == {"prompt", "output"}


@pytest.mark.unit
def test_alpaca_tokenize_normal(alpaca_handler, alpaca_sample_data):
    """Test tokenization with valid data"""
    tokenized = alpaca_handler.tokenize_function(alpaca_sample_data)

    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert "labels" in tokenized
    assert len(tokenized["input_ids"]) == 2
    assert len(tokenized["input_ids"][0]) == 256


@pytest.mark.unit
def test_alpaca_tokenize_malformed(alpaca_handler):
    """Test tokenization error handling for malformed data"""
    malformed_data = {
        "prompt": ["Prompt without output marker"],
        "output": [None],  # Invalid type
    }

    with pytest.raises(TypeError):
        alpaca_handler.tokenize_function(malformed_data)


@pytest.mark.unit
def test_alpaca_prepare_data(alpaca_handler, mocker):
    """Test end-to-end data preparation"""
    mock_load = mocker.patch("datasets.load_dataset")
    mock_data = Dataset.from_dict(
        {"prompt": ["test1", "test2"], "output": ["out1", "out2"]}
    )
    mock_load.return_value = mock_data
    print(f"DEBUG: Mock return type: {type(mock_data)}")  # Add debug log

    prepared = alpaca_handler.prepare_data()
    assert "train" in prepared
    assert "eval" in prepared
    assert prepared["train"].format["type"] == "torch"


@pytest.mark.parametrize(
    "handler_class",
    [
        # GitHubCodeHandler,
        # TinyStoriesHandler,
        # WikiTextHandler,
        PythonAlpacaHandler
    ],
)
@pytest.mark.unit
def test_handler_interface_compliance(handler_class, mock_tokenizer):
    """Test all handlers implement required interface"""
    handler = handler_class(mock_tokenizer)

    assert hasattr(handler, "load_data")
    assert hasattr(handler, "tokenize_function")
    assert hasattr(handler, "prepare_data")

    # Verify method signatures
    assert callable(handler.load_data)
    assert callable(handler.tokenize_function)
    assert callable(handler.prepare_data)


@pytest.mark.unit
def test_base_class_prepare_data(mocker, mock_tokenizer):
    """Test base class prepare_data method structure"""

    class ConcreteHandler(BaseDatasetHandler):
        def load_default_data(self, **kwargs):
            return {
                "train": Dataset.from_dict({"text": ["sample"]}),
                "test": Dataset.from_dict({"text": ["test"]}),
            }

        def tokenize_function(self, examples):
            return {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "labels": [[1, 2, 3]],
            }

    handler = ConcreteHandler(mock_tokenizer)
    prepared = handler.prepare_data()

    assert "train" in prepared
    assert "eval" in prepared
    assert prepared["train"].format["type"] == "torch"
