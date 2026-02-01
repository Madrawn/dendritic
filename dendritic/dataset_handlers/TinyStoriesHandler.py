"""
TinyStoriesHandler - concrete handler for TinyStories dataset.
"""

from typing import Any, Dict, Optional
from transformers.tokenization_utils import PreTrainedTokenizer

from dendritic.dataset_handlers.TextCorpusHandler import TextCorpusHandler


class TinyStoriesHandler(TextCorpusHandler):
    """Handler for TinyStories dataset (roneneldan/TinyStories)."""

    dataset_name = "roneneldan/TinyStories"
    text_column = "text"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        dataset_name: str = "",
        text_column: str = "text",
    ):
        # If not overridden, use class defaults
        if not dataset_name:
            dataset_name = self.dataset_name
        if not text_column:
            text_column = self.text_column
        super().__init__(
            tokenizer,
            max_length,
            dataset_name=dataset_name,
            text_column=text_column,
        )

    def load_default_data(
        self,
        max_samples: int = 10,
        split: str = "train",
        test_size: float = 0.1,
        seed: int = 42,
        streaming: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Load TinyStories dataset with proper configuration.

        The dataset has predefined train/validation splits, so we load them directly
        instead of splitting in memory when split='train' and test_size > 0.
        """
        # TinyStories has predefined splits: 'train' and 'validation'
        # If user requests 'train' split with test_size > 0, we can either:
        # 1. Load 'train' and split it (current super() behavior)
        # 2. Load 'train' for training and 'validation' for testing
        # We'll implement option 2 for better alignment with dataset design

        if split == "train" and test_size > 0:
            # Load train split for training data
            train_ds = super().load_default_data(
                max_samples=max_samples,
                split="train",
                test_size=0.0,  # No split, we'll handle validation separately
                seed=seed,
                streaming=streaming,
                **kwargs,
            )["train"]

            # Load validation split for testing data
            # Calculate validation samples based on test_size ratio
            val_samples = int(max_samples * test_size)
            if val_samples < 1:
                val_samples = 1

            val_ds = super().load_default_data(
                max_samples=val_samples,
                split="validation",
                test_size=0.0,  # No split
                seed=seed,
                streaming=streaming,
                **kwargs,
            )[
                "train"
            ]  # load_default_data returns dict with 'train' key when test_size=0

            return {"train": train_ds, "test": val_ds}
        else:
            # For other cases, use parent implementation
            return super().load_default_data(
                max_samples=max_samples,
                split=split,
                test_size=test_size,
                seed=seed,
                streaming=streaming,
                **kwargs,
            )
