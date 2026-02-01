"""
OpenWebMathHandler - concrete handler for OpenWebMath dataset.
"""

from typing import Any, Dict, Optional
from transformers.tokenization_utils import PreTrainedTokenizer

from dendritic.dataset_handlers.TextCorpusHandler import TextCorpusHandler


class OpenWebMathHandler(TextCorpusHandler):
    """Handler for OpenWebMath dataset (open-web-math/open-web-math)."""

    dataset_name = "open-web-math/open-web-math"
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
