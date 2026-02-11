# ruff: noqa: PLR6301, PLR2004

"""
Unit tests for doubt experiment data loader.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from torch.utils.data import DataLoader, Dataset

from dendritic.experiments.doubt.config import DoubtExperimentConfig
from dendritic.experiments.doubt.data_loader import (
    prepare_doubt_data,
    _create_modified_config,
    _convert_to_doubt_loader,
    create_doubt_batch_from_sequences,
)
from dendritic.experiments.utils.PretrainingConfig import PretrainingConfig


# Mock handler that returns small dataset
class SmallDataset(Dataset):
    def __init__(self, config):
        self.config = config

    def __len__(self):
        return self.config.batch_size * 2  # Enough for 2 full batches

    def __getitem__(self, idx):
        # Return sequences of length max_seq_len + 1 (for lookahead)
        seq_len = self.config.max_seq_len + 1
        return {
            "input_ids": torch.randint(0, 1000, (seq_len,)),
            "labels": torch.randint(0, 1000, (seq_len,)),
        }


class SmallHandler:
    def __init__(self, *args, **kwargs):
        pass

    def prepare_pretraining_dataloaders(self, config, num_workers=0, *args, **kwargs):
        dataset = SmallDataset(config)
        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,  # Ensure consistent batch sizes
        )
        eval_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
        return {"train": train_loader, "eval": eval_loader}


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab_size = 10000

    def __call__(
        self,
        texts,
        truncation=False,
        padding=False,
        max_length=None,
        return_tensors=None,
        *args,
        **kwargs,
    ):
        # Simple tokenization: return sequential IDs

        if isinstance(texts, str):
            texts = [texts]
        tokenized = []
        for text in texts:
            # Generate random token IDs for testing
            length = min(len(text.split()), max_length) if max_length else len(text.split())
            tokens = list(range(100, 100 + length))
            tokenized.append(tokens)

        class MockBatchEncoding(dict):
            def __getattr__(self, attr):
                try:
                    return self[attr]
                except KeyError:
                    raise AttributeError(f"'MockBatchEncoding' has no attribute '{attr}'")

        return MockBatchEncoding({"input_ids": tokenized})


# Create a simple mock dataset
class MockDataset(Dataset):
    def __init__(self, seq_len, batch_size):
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        # Return sequences of length seq_len + 1 (for lookahead)
        input_ids = torch.randint(0, 1000, (self.seq_len + 1,))
        labels = torch.randint(0, 1000, (self.seq_len + 1,))
        return {"input_ids": input_ids, "labels": labels}


class MockDatasetHandler:
    """Mock dataset handler for testing."""

    def __init__(self, tokenizer, max_length=1024, **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_pretraining_dataloaders(self, config, num_workers=0, *args, **kwargs):
        """Return mock dataloaders."""

        train_dataset = MockDataset(config.max_seq_len, config.batch_size)
        eval_dataset = MockDataset(config.max_seq_len, config.batch_size)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,  # Ensure all batches have same size
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,  # Ensure all batches have same size
        )

        return {"train": train_loader, "eval": eval_loader}


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def doubt_config():
    """Create a DoubtExperimentConfig for testing."""
    return DoubtExperimentConfig(
        max_seq_len=128,
        batch_size=4,
        dataset="test_dataset",
        training_steps=100,
        eval_split_ratio=0.1,
        doubt_alpha=1.0,
    )


@pytest.mark.unit
def test_create_modified_config(doubt_config):
    """Test that _create_modified_config increases sequence length by 1."""
    modified = _create_modified_config(doubt_config)

    # Should be a PretrainingConfig (not DoubtExperimentConfig)
    assert isinstance(modified, PretrainingConfig)
    assert not isinstance(modified, DoubtExperimentConfig)

    # Sequence length should be increased by 1
    assert modified.max_seq_len == doubt_config.max_seq_len + 1

    # Other fields should be copied
    assert modified.vocab_size == doubt_config.vocab_size
    assert modified.embed_dim == doubt_config.embed_dim
    assert modified.batch_size == doubt_config.batch_size
    assert modified.dataset == doubt_config.dataset


@pytest.mark.unit
def test_create_doubt_batch_from_sequences():
    """Test the create_doubt_batch_from_sequences function."""
    batch_size = 3
    seq_len = 5
    total_len = seq_len + 1

    # Create input tensor of shape (batch_size, seq_len + 1)
    input_ids = torch.randint(0, 1000, (batch_size, total_len))

    # Extract pairs
    tokens_t, tokens_t_plus_1 = create_doubt_batch_from_sequences(input_ids, seq_len)

    # Check shapes
    assert tokens_t.shape == (batch_size, seq_len)
    # tokens_t_plus_1 is a single token (not a sequence)
    assert tokens_t_plus_1.shape == (batch_size,)

    # Check content
    for i in range(batch_size):
        # tokens_t should be first seq_len tokens
        assert torch.all(tokens_t[i] == input_ids[i, :seq_len])
        # tokens_t_plus_1 should be single token at position seq_len
        assert torch.all(tokens_t_plus_1[i] == input_ids[i, seq_len])


@pytest.mark.unit
def test_create_doubt_batch_from_sequences_errors():
    """Test error cases for create_doubt_batch_from_sequences."""
    # Test with wrong dimensions
    with pytest.raises(ValueError, match="Expected 2D tensor"):
        create_doubt_batch_from_sequences(torch.randn(3, 4, 5), seq_len=3)

    # Test with insufficient length
    input_ids = torch.randint(0, 1000, (2, 3))  # length 3
    with pytest.raises(ValueError, match="Sequence length 3 is less than required"):
        create_doubt_batch_from_sequences(input_ids, seq_len=3)  # needs 3+1=4


@pytest.mark.unit
def test_convert_to_doubt_loader():
    """Test _convert_to_doubt_loader function."""
    batch_size = 2
    seq_len = 8
    total_len = seq_len + 1

    # Create a mock dataloader that yields batches with input_ids
    class MockDataset(Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, idx):
            return {"input_ids": torch.randint(0, 1000, (total_len,))}

    dataset = MockDataset()
    original_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    # Convert to doubt loader
    doubt_loader = _convert_to_doubt_loader(original_loader, seq_len, batch_size)

    # Check that we get pairs
    for batch in doubt_loader:
        tokens_t, tokens_t_plus_1 = batch

        # Check shapes
        assert tokens_t.shape == (batch_size, seq_len)
        # tokens_t_plus_1 is a single token (not a sequence)
        assert tokens_t_plus_1.shape == (batch_size,)

        # Check that they're properly shifted
        # Since we can't access the original input_ids after conversion,
        # we just verify the shapes and types
        assert isinstance(tokens_t, torch.Tensor)
        assert isinstance(tokens_t_plus_1, torch.Tensor)
        break  # Just test first batch


@pytest.mark.unit
@patch("dendritic.experiments.doubt.data_loader.get_handler")
def test_prepare_doubt_data_mock(mock_get_handler, doubt_config, mock_tokenizer):
    """Test prepare_doubt_data with mocked handler."""
    # Setup mock handler with mocked prepare_pretraining_dataloaders
    mock_handler = Mock(spec=MockDatasetHandler)

    # Create a simple mock dataset that returns proper data
    class MockDataset(Dataset):
        def __init__(self, seq_len):
            self.seq_len = seq_len

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            # Return sequences of length seq_len + 1
            input_ids = torch.randint(0, 1000, (self.seq_len + 1,))
            return {"input_ids": input_ids}

    # Create real dataloaders with mock dataset
    seq_len = doubt_config.max_seq_len
    mock_dataset = MockDataset(seq_len)
    mock_train_loader = DataLoader(
        mock_dataset,
        batch_size=doubt_config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    mock_eval_loader = DataLoader(
        mock_dataset,
        batch_size=doubt_config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    mock_handler.prepare_pretraining_dataloaders.return_value = {
        "train": mock_train_loader,
        "eval": mock_eval_loader,
    }

    mock_get_handler.return_value = mock_handler

    # Call prepare_doubt_data
    loaders = prepare_doubt_data(
        config=doubt_config,
        tokenizer=mock_tokenizer,
        dataset_kwargs={"max_samples": 100},
        num_workers=0,
    )

    # Check that get_handler was called with correct arguments
    mock_get_handler.assert_called_once()
    call_args = mock_get_handler.call_args
    assert call_args[0][0] == doubt_config.dataset
    assert call_args[0][1] == mock_tokenizer
    assert call_args[1]["max_length"] == doubt_config.max_seq_len + 1
    # max_samples should NOT be passed to get_handler, it should be passed to prepare_pretraining_dataloaders
    assert "max_samples" not in call_args[1]

    # Check that prepare_pretraining_dataloaders was called
    mock_handler.prepare_pretraining_dataloaders.assert_called_once()

    # Check returned loaders (they should be converted doubt loaders, not the mock ones)
    assert "train" in loaders
    assert "eval" in loaders
    assert isinstance(loaders["train"], DataLoader)
    assert isinstance(loaders["eval"], DataLoader)

    # Check that doubt loaders yield pairs
    for loader_name, loader in loaders.items():
        for batch in loader:
            tokens_t, tokens_t_plus_1 = batch
            assert tokens_t.shape == (
                doubt_config.batch_size,
                doubt_config.max_seq_len,
            )
            # tokens_t_plus_1 is a single token (not a sequence)
            assert tokens_t_plus_1.shape == (doubt_config.batch_size,)
            break  # Just test first batch


@pytest.mark.unit
@patch("dendritic.experiments.doubt.data_loader.get_handler")
def test_prepare_doubt_data_small_dataset(mock_get_handler, doubt_config, mock_tokenizer):
    """Test prepare_doubt_data with small dataset."""

    mock_get_handler.return_value = SmallHandler()

    # This should work
    loaders = prepare_doubt_data(config=doubt_config, tokenizer=mock_tokenizer, num_workers=0)

    assert "train" in loaders
    assert "eval" in loaders
    # Check that loaders produce batches
    train_batch = next(iter(loaders["train"]))
    assert isinstance(train_batch, tuple)
    assert len(train_batch) == 2


@pytest.mark.unit
def test_prepare_doubt_data_integration(doubt_config, mock_tokenizer):
    """Integration test for prepare_doubt_data with actual handler patching."""
    # We'll patch the handler factory to return our mock handler
    with patch("dendritic.experiments.doubt.data_loader.get_handler") as mock_get:
        mock_handler = MockDatasetHandler(mock_tokenizer)
        mock_get.return_value = mock_handler

        loaders = prepare_doubt_data(config=doubt_config, tokenizer=mock_tokenizer, num_workers=0)

        # Verify the loaders produce the expected format
        train_loader = loaders["train"]
        first_batch = next(iter(train_loader))

        # Should be a tuple of 2 tensors
        assert isinstance(first_batch, tuple)
        assert len(first_batch) == 2

        tokens_t, tokens_t_plus_1 = first_batch
        assert tokens_t.shape == (
            doubt_config.batch_size,
            doubt_config.max_seq_len,
        )
        # tokens_t_plus_1 is a single token (not a sequence)
        assert tokens_t_plus_1.shape == (doubt_config.batch_size,)


@pytest.mark.unit
def test_doubt_dataset_wrapper():
    """Test the internal DoubtDataset class."""
    # This tests the internal class by calling _convert_to_doubt_loader
    batch_size = 2
    seq_len = 10
    total_len = seq_len + 1

    # Create a simple dataloader
    class SimpleDataset(Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            return {"input_ids": torch.arange(idx * 100, idx * 100 + total_len, dtype=torch.long)}

    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Convert to doubt loader
    doubt_loader = _convert_to_doubt_loader(dataloader, seq_len, batch_size=1, shuffle=False)

    # Check all batches
    for i, (tokens_t, tokens_t_plus_1) in enumerate(doubt_loader):
        # Each batch should have shape (1, seq_len) for tokens_t, single token for tokens_t_plus_1
        assert tokens_t.shape == (1, seq_len)
        assert tokens_t_plus_1.shape == (1,)

        # Check shifting
        expected_start = i * 100
        expected_t = torch.arange(expected_start, expected_start + seq_len, dtype=torch.long)
        # tokens_t_plus_1 is single token at position seq_len
        expected_t1 = torch.tensor([expected_start + seq_len], dtype=torch.long)

        assert torch.all(tokens_t[0] == expected_t)
        assert torch.all(tokens_t_plus_1[0] == expected_t1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
