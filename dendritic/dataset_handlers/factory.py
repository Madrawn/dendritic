"""
Dataset handler factory and registry.
"""

from typing import Dict, Type, Any, Optional
from transformers.tokenization_utils import PreTrainedTokenizer

from dendritic.dataset_handlers.BaseDatasetHandler import BaseDatasetHandler


# Global registry mapping dataset name to handler class
_REGISTRY: Dict[str, Type[BaseDatasetHandler]] = {}


def register_handler(name: str, handler_cls: Type[BaseDatasetHandler]) -> None:
    """Register a handler class under the given dataset name."""
    if name in _REGISTRY:
        raise ValueError(f"Dataset handler '{name}' already registered")
    _REGISTRY[name] = handler_cls


def get_handler(
    name: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    **kwargs,
) -> BaseDatasetHandler:
    """
    Instantiate a handler for the given dataset name.
    
    Parameters
    ----------
    name : str
        Dataset identifier (e.g., 'wikitext', 'openwebmath', 'python_alpaca').
    tokenizer : PreTrainedTokenizer
        Tokenizer to pass to the handler.
    max_length : int, optional
        Maximum sequence length.
    **kwargs : dict
        Additional keyword arguments passed to the handler's constructor.
        
    Returns
    -------
    BaseDatasetHandler
        Instantiated handler.
        
    Raises
    ------
    KeyError
        If `name` is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"No dataset handler registered for '{name}'. "
                       f"Available handlers: {list(_REGISTRY.keys())}")
    handler_cls = _REGISTRY[name]
    return handler_cls(tokenizer, max_length=max_length, **kwargs)


def list_handlers() -> Dict[str, Type[BaseDatasetHandler]]:
    """Return a copy of the registry."""
    return _REGISTRY.copy()


# Auto‑register default handlers
try:
    from dendritic.dataset_handlers.WikiTextHandler import WikiTextHandler
    register_handler("wikitext", WikiTextHandler)
except ImportError:
    pass

try:
    from dendritic.dataset_handlers.OpenWebMathHandler import OpenWebMathHandler
    register_handler("openwebmath", OpenWebMathHandler)
except ImportError:
    pass

try:
    from dendritic.dataset_handlers.PythonAlpacaHandler import PythonAlpacaHandler
    register_handler("python_alpaca", PythonAlpacaHandler)
except ImportError:
    pass