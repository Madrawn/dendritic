from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


import os
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseDatasetHandler(ABC):
    """Abstract base class for dataset handlers."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        text_column: str = "text",
        pad_token_strategy: Optional[str] = "auto",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.dataset_name = ""  # Should be set by subclasses
        self.predefined_splits = None  # Should be set by subclasses

        # Configure pad token with flexible strategy
        self._configure_pad_token(pad_token_strategy)

    def _configure_pad_token(self, strategy: Optional[str] = "auto") -> None:
        """Configure pad token with flexible strategy.

        Parameters
        ----------
        strategy : Optional[str]
            Strategy for pad token configuration. Options:
            - "auto": Use existing pad_token if available, otherwise use eos_token
            - "eos": Always set pad_token to eos_token
            - "custom": Use tokenizer's default pad_token (no modification)
            - None: Don't configure pad_token
        """
        if strategy == "auto":
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif strategy == "eos":
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif strategy == "custom":
            # Use tokenizer's default pad_token, no modification
            pass
        elif strategy is None:
            # Don't configure pad_token
            pass
        else:
            raise ValueError(f"Unknown pad_token_strategy: {strategy}. Expected 'auto', 'eos', 'custom', or None.")

    def load_data(self, **kwargs) -> dict[str, Any]:
        """Load and return the dataset splits.
        If 'data_files' is provided, load from specified file(s).
        Otherwise, call the abstract method `load_default_data`.
        """
        if "data_files" in kwargs:
            # Extract data_files and remove from kwargs to avoid duplication
            data_files = kwargs.pop("data_files")
            return self.load_from_file(data_files, **kwargs)
        else:
            return self.load_default_data(**kwargs)

    def load_from_file(self, data_files, **kwargs) -> dict[str, Any]:
        """Load dataset from file(s)."""
        from datasets import load_dataset
        import os

        # Determine file format based on extension
        if isinstance(data_files, str):
            ext = os.path.splitext(data_files)[1].lower()
        elif isinstance(data_files, (list, tuple)) and data_files:
            ext = os.path.splitext(data_files[0])[1].lower()
        else:
            ext = ".json"  # default to JSON

        if ext == ".csv":
            dataset = load_dataset("csv", data_files=data_files)
        elif ext == ".txt":
            dataset = load_dataset("text", data_files=data_files)
        else:
            dataset = load_dataset("json", data_files=data_files)

        # Return the dataset without splitting
        return {
            "train": dataset["train"],
            "test": Dataset.from_dict({}),  # empty test set
        }

    @abstractmethod
    def load_default_data(self, *args, **kwargs) -> dict[str, Any]:
        """Load the default dataset when no data_files are provided."""
        pass

    @abstractmethod
    def tokenize_function(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Tokenize batch of examples with proper masking."""
        pass

    def prepare_pretraining_dataloaders(self, config: Any, num_workers: int = 0, *args, **kwargs) -> dict[str, Any]:
        """Prepare dataloaders for pretraining."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement pretraining dataloader preparation. "
            "Use a dataset handler that inherits from TextCorpusHandler."
        )

    def tokenize_for_pretraining(self, examples: dict[str, Any], append_newline: bool = True) -> dict[str, Any]:
        """
        Tokenize a batch of text examples for pretraining (no padding, no truncation).

        This is used before block concatenation. Optionally appends a newline token
        to separate documents/sentences.

        The base implementation raises NotImplementedError; subclasses that support
        pretraining should override this method.

        Parameters
        ----------
        examples : dict[str, Any]
            Batch with at least the key `self.text_column` containing a list of strings.
        append_newline : bool
            Whether to append a newline token to each text.

        Returns
        -------
        dict[str, Any]
            Tokenized batch with keys 'input_ids' (list of token IDs).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support pretraining tokenization. "
            "Use a dataset handler that inherits from TextCorpusHandler."
        )

    def prepare_data(self, **kwargs) -> dict[str, Any]:
        """Load and prepare the dataset, including tokenization."""
        ds = self.load_data(**kwargs)

        # Tokenize datasets
        train_dataset = ds["train"].map(
            self.tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=ds["train"].column_names,
        )

        # If test set exists, tokenize it
        if len(ds["test"]) > 0:
            eval_dataset = ds["test"].map(
                self.tokenize_function,
                batched=True,
                batch_size=100,
                remove_columns=ds["test"].column_names,
            )
        else:
            eval_dataset = Dataset.from_dict({})

        # Set format for PyTorch
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        if len(eval_dataset) > 0:
            eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        return {"train": train_dataset, "eval": eval_dataset}

    def validate_tokenization_inputs(self, examples: dict[str, Any]) -> None:
        """Validate inputs before tokenization.

        Parameters
        ----------
        examples : dict[str, Any]
            Input examples to validate.

        Raises
        ------
        ValueError
            If required columns are missing or have invalid data types.
        """
        if self.text_column not in examples:
            raise TypeError(f"Missing required column '{self.text_column}' in examples")

        if not isinstance(examples[self.text_column], list):
            raise TypeError(f"Column '{self.text_column}' must be a list")

        if not all(isinstance(text, str) for text in examples[self.text_column]):
            raise TypeError(f"All values in '{self.text_column}' must be strings")

        # Check for empty strings
        if any(text.strip() == "" for text in examples[self.text_column]):
            raise TypeError(f"Found empty strings in '{self.text_column}' column")

    def validate_instruction_inputs(self, examples: dict[str, Any]) -> None:
        """Validate inputs for instruction-following datasets.

        Parameters
        ----------
        examples : dict[str, Any]
            Input examples to validate.

        Raises
        ------
        ValueError
            If required columns are missing or have invalid data types.
        """
        # Use configurable column names if set by subclass, otherwise fall back to defaults
        prompt_column = getattr(self, "prompt_column", "prompt")
        response_column = getattr(self, "response_column", "output")

        required_columns = [prompt_column, response_column]
        for column in required_columns:
            if column not in examples:
                raise TypeError(f"Missing required column '{column}' in examples")

            if not isinstance(examples[column], list):
                raise TypeError(f"Column '{column}' must be a list")

            if not all(isinstance(item, str) for item in examples[column]):
                raise TypeError(f"All values in '{column}' must be strings")

            # Check for None values
            if any(item is None for item in examples[column]):
                raise TypeError(f"Column '{column}' contains None values")

            # Check for empty strings
            if any(item.strip() == "" for item in examples[column]):
                raise TypeError(f"Found empty strings in '{column}' column")

    def validate_tokenizer_config(self) -> None:
        """Validate tokenizer configuration.

        Raises
        ------
        ValueError
            If tokenizer is not properly configured.
        """
        if not hasattr(self.tokenizer, "encode"):
            raise ValueError("Tokenizer must have 'encode' method")

        if not hasattr(self.tokenizer, "decode"):
            raise ValueError("Tokenizer must have 'decode' method")

        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")

        # Test basic tokenization
        test_text = "Hello world"
        try:
            tokens = self.tokenizer.encode(test_text)
            if not isinstance(tokens, list):
                raise ValueError("Tokenizer.encode() must return a list")
            if not all(isinstance(token, int) for token in tokens):
                raise ValueError("All tokens must be integers")
        except Exception as e:
            raise ValueError(f"Tokenizer test failed: {e}")
