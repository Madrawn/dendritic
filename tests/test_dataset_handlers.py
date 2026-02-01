import pytest
import dendritic
from dendritic.dataset_handlers.BaseDatasetHandler import BaseDatasetHandler
from dendritic.dataset_handlers.PythonAlpacaHandler import PythonAlpacaHandler
from dendritic.dataset_handlers.dataset_handlers import (
    WikiTextHandler,
)
from transformers.models.gpt2 import GPT2Tokenizer

from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import Dataset, IterableDataset
import torch

import dendritic.experiments
import dendritic.experiments.utils
import dendritic.experiments.utils.PretrainingConfig
from dendritic.experiments.run_experiments import calculate_required_max_samples
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
    mock_load = mocker.patch(
        "dendritic.dataset_handlers.TextCorpusHandler.load_dataset"
    )
    mock_load.return_value = Dataset.from_dict({"text": ["sample1", "sample2"] * 100})
    data = wikitext_handler.load_data(split="train", test_size=0.5, max_samples=200)
    assert "train" in data
    assert "test" in data
    assert len(data["train"]) == 100
    assert set(data["train"].column_names) == {"text"}


@pytest.mark.unit
def test_wikitext_tokenize_malformed(wikitext_handler):
    """Test WikiText tokenization error handling for malformed data"""
    malformed_data = {"text": [None]}
    with pytest.raises(Exception):
        wikitext_handler.tokenize_function(malformed_data)


def sample_generator():
    long_text = "sample text " * 1000
    for _ in range(200):
        yield {"text": long_text}


@pytest.mark.integration
# @pytest.mark.skip(reason="TODO")
def test_wikitext_prepare_data(mock_tokenizer: PreTrainedTokenizer, mocker):
    """Test WikiText end-to-end data preparation"""
    handler = WikiTextHandler(mock_tokenizer, max_length=5000)
    mock_load = mocker.patch(
        "dendritic.dataset_handlers.TextCorpusHandler.load_dataset"
    )
    # Use longer samples to ensure they don't get filtered out or result in empty blocks
    mock_data = IterableDataset.from_generator(sample_generator)
    mock_load.return_value = mock_data
    cfg = dendritic.experiments.utils.PretrainingConfig(
        training_steps=300,
        batch_size=2,
        max_seq_len=256,
        eval_split_ratio=0.5,
        grouped=True,
        group_separator="EOS_token",
    )
    required_max_samples = calculate_required_max_samples(cfg)
    cfg.dataset_kwargs = {"max_samples": required_max_samples}
    prepared = handler.prepare_pretraining_dataloaders(
        config=cfg, num_workers=0, kwargs=cfg.dataset_kwargs
    )

    assert "train" in prepared
    assert "eval" in prepared
    train_length = sum(1 for _ in prepared["train"])
    # with training_steps=300, batch_size=2, eval_split_ratio=0.5 we expect samples
    assert train_length >= int(300)
    # For DataLoader, we check format on the underlying dataset
    # Instead of checking .format["type"]
    # We fetch one batch and verify the content is a torch.Tensor
    first_batch = next(iter(prepared["train"]))

    assert isinstance(first_batch["input_ids"], torch.Tensor)
    assert isinstance(first_batch["labels"], torch.Tensor)
    assert first_batch["input_ids"].shape == (cfg.batch_size, cfg.max_seq_len)


@pytest.mark.integration
# @pytest.mark.skip(reason="TODO")
def test_wikitext_prepare_data_ungrouped(mocker):
    """Test WikiText end-to-end data preparation with grouped=False."""

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    handler = WikiTextHandler(tokenizer, max_length=5000)
    mock_load = mocker.patch(
        "dendritic.dataset_handlers.TextCorpusHandler.load_dataset"
    )
    # Use longer samples to ensure they don't get filtered out or result in empty blocks
    mock_data = IterableDataset.from_generator(sample_generator)
    mock_load.return_value = mock_data
    training_steps_count = 25
    cfg = dendritic.experiments.utils.PretrainingConfig(
        training_steps=training_steps_count,
        batch_size=2,
        max_seq_len=256,
        eval_split_ratio=0.5,
        grouped=False,
        group_separator="EOS_token",
    )
    required_max_samples = calculate_required_max_samples(cfg)
    cfg.dataset_kwargs = {"max_samples": required_max_samples}
    prepared = handler.prepare_pretraining_dataloaders(
        config=cfg, kwargs=cfg.dataset_kwargs
    )
    assert "train" in prepared
    assert "eval" in prepared
    train_length = sum(1 for _ in prepared["train"])
    # with training_steps=5, batch_size=2, eval_split_ratio=0.5 we expect samples
    assert train_length >= int(training_steps_count)

    # For DataLoader, we check format on the underlying dataset
    # Instead of checking .format["type"]
    # Additionally, verify that each sample is padded to max_seq_len
    batch = next(iter(prepared["train"]))
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["labels"], torch.Tensor)
    assert batch["input_ids"].shape == (cfg.batch_size, cfg.max_seq_len)

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    assert input_ids.shape == (2, 256)
    assert labels.shape == (2, 256)
    # Check padding masking
    pad_token_id = handler.tokenizer.pad_token_id
    if pad_token_id is not None:
        pad_mask = input_ids == pad_token_id
        label_mask = labels == -100
        assert (
            pad_mask == label_mask
        ).all(), "Padding tokens should be masked in labels"


@pytest.fixture
def wikitext_handler():
    return WikiTextHandler(GPT2Tokenizer.from_pretrained("gpt2"), max_length=2048)


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
