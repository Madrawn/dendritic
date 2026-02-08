"""
Dataset handlers for various corpora.
"""

from .BaseDatasetHandler import BaseDatasetHandler
from .InstructionHandler import InstructionHandler
from .PythonAlpacaHandler import PythonAlpacaHandler
from .dataset_handlers import (
    TinyStoriesHandler,
    WikiTextHandler,
    OpenWebMathHandler,
)
from .TextCorpusHandler import TextCorpusHandler
from .factory import register_handler, get_handler, list_handlers

__all__ = [
    "BaseDatasetHandler",
    "InstructionHandler",
    "PythonAlpacaHandler",
    "TinyStoriesHandler",
    "WikiTextHandler",
    "OpenWebMathHandler",
    "TextCorpusHandler",
    "register_handler",
    "get_handler",
    "list_handlers",
]
