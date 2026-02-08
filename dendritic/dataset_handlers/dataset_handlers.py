from typing import Dict, Any, List

from .BaseDatasetHandler import BaseDatasetHandler
from .WikiTextHandler import WikiTextHandler
from .OpenWebMathHandler import OpenWebMathHandler
from .TinyStoriesHandler import TinyStoriesHandler


# Export the real handlers
__all__ = [
    "TinyStoriesHandler",
    "WikiTextHandler",
    "OpenWebMathHandler",
]
