"""
Integration tests for sliding window functionality with mocked datasets.
Tests the end-to-end flow from dataset loading through tokenization to dataloader creation.
"""

import pytest
import torch
from transformers import AutoTokenizer
from dendritic.dataset_handlers.dataset_handlers import WikiTextHandler
from dendritic.experiments.utils.PretrainingConfig import PretrainingConfig
from dendritic.experiments.run_experiments import calculate_required_max_samples
from datasets import IterableDataset


def number_generator():
    """Generate samples with predictable token sequences."""
    for i in range(20):
        text = " ".join(str(j) for j in range(1, 11)) * 2
        yield {"text": text}


@pytest.mark.integration
class TestSlidingWindowIntegration:
    """Test sliding window with actual dataset preparation pipeline."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a simple tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @pytest.fixture
    def wikitext_handler(self, mock_tokenizer):
        """Create a WikiTextHandler for testing."""
        return WikiTextHandler(mock_tokenizer, max_length=5000)

    @pytest.mark.integration
    def test_sliding_window_with_grouped_dataset(self, mocker, wikitext_handler, mock_tokenizer):
        """Test that sliding window creates overlapping sequences in grouped dataset."""
        # Mock the dataset loading
        mock_load = mocker.patch("dendritic.dataset_handlers.TextCorpusHandler.load_dataset")
        mock_data = IterableDataset.from_generator(number_generator)
        mock_load.return_value = mock_data

        # Create config with seq_stride=1
        cfg = PretrainingConfig(
            training_steps=100,
            batch_size=2,
            max_seq_len=3,
            eval_split_ratio=0.5,
            grouped=True,
            group_separator="EOS_token",
            seq_stride=1,
        )
        required_max_samples = calculate_required_max_samples(cfg)
        cfg.dataset_kwargs = {"max_samples": required_max_samples}

        # Prepare dataloaders
        prepared = wikitext_handler.prepare_pretraining_dataloaders(config=cfg, kwargs=cfg.dataset_kwargs)

        # Collect all sequences from train dataloader
        all_input_ids = []
        for batch in prepared["train"]:
            all_input_ids.extend(batch["input_ids"].tolist())

        # Verify we have sequences
        assert len(all_input_ids) > 0

        # Check that all sequences have correct length
        for seq in all_input_ids:
            assert len(seq) == 3

        # The sequences should be overlapping. For a long concatenated sequence,
        # with max_seq_len=3 and seq_stride=1, we get:
        # [t1,t2,t3], [t2,t3,t4], [t3,t4,t5], ...
        # We can verify the pattern by checking consecutive sequences
        if len(all_input_ids) >= 2:
            # First two sequences should overlap by 2 tokens
            seq1 = all_input_ids[0]
            seq2 = all_input_ids[1]
            assert seq1[1] == seq2[0], "Sequences should overlap"
            assert seq1[2] == seq2[1], "Sequences should overlap by 2 tokens"

    @pytest.mark.integration
    def test_sliding_window_stride_2(self, mocker, wikitext_handler, mock_tokenizer):
        """Test sliding window with stride=2 creates less overlap."""
        mock_load = mocker.patch("dendritic.dataset_handlers.TextCorpusHandler.load_dataset")
        mock_data = IterableDataset.from_generator(number_generator)
        mock_load.return_value = mock_data

        cfg = PretrainingConfig(
            training_steps=100,
            batch_size=2,
            max_seq_len=3,
            eval_split_ratio=0.5,
            grouped=True,
            group_separator="EOS_token",
            seq_stride=2,
        )
        required_max_samples = calculate_required_max_samples(cfg)
        cfg.dataset_kwargs = {"max_samples": required_max_samples}

        prepared = wikitext_handler.prepare_pretraining_dataloaders(config=cfg, kwargs=cfg.dataset_kwargs)

        all_input_ids = []
        for batch in prepared["train"]:
            all_input_ids.extend(batch["input_ids"].tolist())

        assert len(all_input_ids) > 0

        # Check all sequences have correct length
        for seq in all_input_ids:
            assert len(seq) == 3

        # With stride=2, consecutive sequences should overlap by 1 token
        if len(all_input_ids) >= 2:
            seq1 = all_input_ids[0]
            seq2 = all_input_ids[1]
            assert seq1[2] == seq2[0], "Sequences should overlap by 1 token with stride=2"

    @pytest.mark.integration
    def test_sliding_window_stride_0_original_behavior(self, mocker, wikitext_handler, mock_tokenizer):
        """Test that seq_stride=0 preserves original non-overlapping behavior."""
        mock_load = mocker.patch("dendritic.dataset_handlers.TextCorpusHandler.load_dataset")
        mock_data = IterableDataset.from_generator(number_generator)
        mock_load.return_value = mock_data

        cfg = PretrainingConfig(
            training_steps=100,
            batch_size=2,
            max_seq_len=3,
            eval_split_ratio=0.5,
            grouped=True,
            group_separator="EOS_token",
            seq_stride=0,
        )
        required_max_samples = calculate_required_max_samples(cfg)
        cfg.dataset_kwargs = {"max_samples": required_max_samples}

        prepared = wikitext_handler.prepare_pretraining_dataloaders(config=cfg, kwargs=cfg.dataset_kwargs)

        all_input_ids = []
        for batch in prepared["train"]:
            all_input_ids.extend(batch["input_ids"].tolist())

        assert len(all_input_ids) > 0

        # With stride=0, sequences should NOT overlap
        # Consecutive sequences should be adjacent in the original token stream
        if len(all_input_ids) >= 2:
            seq1 = all_input_ids[0]
            seq2 = all_input_ids[1]
            # They should not overlap (no shared tokens)
            assert seq1[1] != seq2[0] or seq1[2] != seq2[1], "Non-overlapping sequences should not share tokens"

    @pytest.mark.integration
    def test_sliding_window_preserves_labels(self, mocker, wikitext_handler, mock_tokenizer):
        """Test that labels are correctly created with sliding window."""
        mock_load = mocker.patch("dendritic.dataset_handlers.TextCorpusHandler.load_dataset")

        def short_generator():
            for i in range(10):
                text = " ".join(str(j) for j in range(1, 6))
                yield {"text": text}

        mock_data = IterableDataset.from_generator(short_generator)
        mock_load.return_value = mock_data

        cfg = PretrainingConfig(
            training_steps=50,
            batch_size=2,
            max_seq_len=3,
            eval_split_ratio=0.5,
            grouped=True,
            group_separator="EOS_token",
            seq_stride=1,
        )
        required_max_samples = calculate_required_max_samples(cfg)
        cfg.dataset_kwargs = {"max_samples": required_max_samples}

        prepared = wikitext_handler.prepare_pretraining_dataloaders(config=cfg, kwargs=cfg.dataset_kwargs)

        # Check that labels match input_ids for language modeling
        for batch in prepared["train"]:
            assert torch.equal(batch["input_ids"], batch["labels"])
        for batch in prepared["eval"]:
            assert torch.equal(batch["input_ids"], batch["labels"])

    @pytest.mark.integration
    def test_sliding_window_batch_structure(self, mocker, wikitext_handler, mock_tokenizer):
        """Test that batches have correct structure and dimensions."""
        mock_load = mocker.patch("dendritic.dataset_handlers.TextCorpusHandler.load_dataset")
        mock_data = IterableDataset.from_generator(number_generator)
        mock_load.return_value = mock_data

        cfg = PretrainingConfig(
            training_steps=100,
            batch_size=4,
            max_seq_len=5,
            eval_split_ratio=0.5,
            grouped=True,
            group_separator="EOS_token",
            seq_stride=1,
        )
        required_max_samples = calculate_required_max_samples(cfg)
        cfg.dataset_kwargs = {"max_samples": required_max_samples}

        prepared = wikitext_handler.prepare_pretraining_dataloaders(config=cfg, kwargs=cfg.dataset_kwargs)

        # Get first batch from train
        train_iter = iter(prepared["train"])
        first_batch = next(train_iter)

        # Check batch dimensions
        assert first_batch["input_ids"].dim() == 2
        assert first_batch["input_ids"].shape[1] == cfg.max_seq_len
        assert first_batch["labels"].shape == first_batch["input_ids"].shape

        # Check types
        assert isinstance(first_batch["input_ids"], torch.Tensor)
        assert isinstance(first_batch["labels"], torch.Tensor)
