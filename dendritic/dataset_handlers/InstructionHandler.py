"""
Base class for instruction-following datasets.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from transformers.tokenization_utils import PreTrainedTokenizer

from .BaseDatasetHandler import BaseDatasetHandler


class InstructionHandler(BaseDatasetHandler, ABC):
    """Base class for instruction-following datasets like PythonAlpaca."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        pad_token_strategy: str = "auto",
    ):
        super().__init__(tokenizer, max_length, "text", pad_token_strategy)
        self.prompt_column = "prompt"
        self.response_column = "output"

    @abstractmethod
    def load_default_data(self, **kwargs) -> dict[str, Any]:
        """Load the default instruction dataset."""
        pass

    def tokenize_function(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Tokenize instruction examples with proper masking.

        This implementation handles the common pattern of tokenizing prompts and responses
        separately and then combining them with appropriate loss masking.
        """
        # Validate inputs
        self.validate_instruction_inputs(examples)

        prompts = examples[self.prompt_column]
        outputs = examples[self.response_column]

        # Tokenize prompts and outputs separately
        prompt_tokens = self.tokenizer(prompts, truncation=False, padding=False, add_special_tokens=False)
        output_texts = []
        for output in outputs:
            eos_token = getattr(self.tokenizer, "eos_token", "") or ""
            output_texts.append(output + eos_token)
        output_tokens = self.tokenizer(output_texts, truncation=False, padding=False, add_special_tokens=False)

        input_ids = []
        labels = []
        max_length = self.max_length
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        for prompt_ids, output_ids in zip(prompt_tokens.input_ids, output_tokens.input_ids):
            full_ids = prompt_ids + output_ids
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            prompt_len = len(prompt_ids)
            full_labels = [-100] * prompt_len + output_ids
            if len(full_labels) > max_length:
                full_labels = full_labels[:max_length]

            padding_length = max_length - len(full_ids)
            full_ids = full_ids + [pad_token_id] * padding_length
            full_labels = full_labels + [-100] * padding_length

            input_ids.append(full_ids)
            labels.append(full_labels)

        attention_mask = [[1 if id != pad_token_id else 0 for id in ids] for ids in input_ids]

        # Validate tokenization results
        self._validate_tokenization_results(input_ids, labels, attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _validate_tokenization_results(
        self, input_ids: list[list[int]], labels: list[list[int]], attention_mask: list[list[int]]
    ) -> None:
        """Validate tokenization results.

        Parameters
        ----------
        input_ids : list[list[int]]
            Input token IDs.
        labels : list[list[int]]
            Label token IDs.
        attention_mask : list[list[int]]
            Attention mask.

        Raises
        ------
        ValueError
            If tokenization results are invalid.
        """
        if len(input_ids) != len(labels) or len(input_ids) != len(attention_mask):
            raise ValueError("input_ids, labels, and attention_mask must have same length")

        for i, (ids, lbls, mask) in enumerate(zip(input_ids, labels, attention_mask)):
            if len(ids) != len(lbls) or len(ids) != len(mask):
                raise ValueError(f"Mismatched lengths at index {i}")

            if len(ids) != self.max_length:
                raise ValueError(f"Sequence length at index {i} is {len(ids)}, expected {self.max_length}")

            # Check for invalid token IDs
            invalid_ids = [x for x in ids if x is None or x < 0 or x > 100000]
            if invalid_ids:
                raise ValueError(f"Invalid token IDs at index {i}: {invalid_ids[:5]}...")

            # Check for invalid label IDs
            invalid_labels = [x for x in lbls if x is None or (x != -100 and (x < 0 or x > 100000))]
            if invalid_labels:
                raise ValueError(f"Invalid label IDs at index {i}: {invalid_labels[:5]}...")

            # Check attention mask
            if not all(m in {0, 1} for m in mask):
                raise ValueError(f"Attention mask contains invalid values at index {i}")
