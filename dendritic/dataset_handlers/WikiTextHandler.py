"""
WikiTextHandler - concrete handler for WikiText-103 dataset.
"""

from typing import Any, Dict, Optional
from transformers.tokenization_utils import PreTrainedTokenizer

from dendritic.dataset_handlers.TextCorpusHandler import TextCorpusHandler


class WikiTextHandler(TextCorpusHandler):
    """Handler for WikiText-103 dataset."""
    
    dataset_name = "wikitext"
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
        max_samples: int=1,
        split: str = "train",
        test_size: float = 0.1,
        seed: int = 42,
        streaming: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Load WikiText-103 dataset with proper configuration.
        
        Overrides to specify the sub‑dataset name "wikitext-103-raw-v1".
        """
        # Use the specific configuration
        kwargs.setdefault("name", "wikitext-103-raw-v1")
        return super().load_default_data(
            max_samples=max_samples,
            split=split,
            test_size=test_size,
            seed=seed,
            streaming=streaming,
            **kwargs,
        )