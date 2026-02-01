"""
Dataset handlers for various corpora.
"""

from .BaseDatasetHandler import BaseDatasetHandler
from .PythonAlpacaHandler import PythonAlpacaHandler
from .dataset_handlers import (
    TinyStoriesHandler,
    GitHubCodeHandler,
    WikiTextHandler,
    OpenWebMathHandler,
)
from .TextCorpusHandler import TextCorpusHandler
from .factory import register_handler, get_handler, list_handlers

__all__ = [
    "BaseDatasetHandler",
    "PythonAlpacaHandler",
    "TinyStoriesHandler",
    "GitHubCodeHandler",
    "WikiTextHandler",
    "OpenWebMathHandler",
    "TextCorpusHandler",
    "register_handler",
    "get_handler",
    "list_handlers",
]
