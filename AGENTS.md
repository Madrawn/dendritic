# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build/Test Commands

- **Run all tests**: `python run_tests.py` (custom test runner with mode selection)
- **Run specific test types**: `python run_tests.py --mode unit|integration|edge|all`
- **Run tests in parallel**: `python run_tests.py --parallel`
- **Run with coverage**: `python run_tests.py --coverage`
- **Run specific test by keyword**: `python run_tests.py -k "test_forward_pass"`

## Project-Specific Patterns

- **Test markers**: Tests use pytest markers (`@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.edge`) - must match conftest.py registration
- **CUDA requirement**: Tests assume CUDA is available (`assert torch.cuda.is_available()` in run_tests.py:6)
- **Experiment structure**: Results use specific model keys ("baseline", "dendritic", "stack", "baseline_wave") with statistical comparisons between them
- **Polynomial rank**: Default poly_rank="auto" computes as `max(4, input_dim // 64)` in enhancement logic
- **Dataset handling**: Text corpora (WikiText, OpenWebMath, etc.) use `TextCorpusHandler` with mandatory `max_samples` (no default), default `streaming=True`, and in‑memory splitting via `Dataset.train_test_split`. The evaluation split ratio can be configured via the `test_size` parameter (default 0.1). `PythonAlpacaHandler` provides prompt formatting for instruction datasets.

## Critical Gotchas

- **Memory management**: Integration tests include explicit garbage collection (`gc.collect()`) due to large model memory requirements
- **Device placement**: Tests verify CUDA/CPU consistency with detailed debugging output when GPU available
- **Serialization**: Numpy types must be converted to native Python types for JSON serialization in experiment results
- **Test data**: Some edge case tests are marked `@pytest.mark.xfail` due to known limitations

## File Organization

- **Experiments**: All experiment code in `dendritic/experiments/` with specific run patterns
- **Layers**: Custom dendritic layers in `dendritic/layers/` with mathematical correctness verification
- **Handlers**: Dataset handlers in `dendritic/dataset_handlers/` with specific interface requirements

## Adding New Datasets

To add a new text corpus dataset:

1. Create a handler class inheriting from `TextCorpusHandler`.
2. Set class attributes `dataset_name` and `text_column`.
3. Optionally override `load_default_data` if the dataset requires special configuration (e.g., sub‑dataset name).
4. The handler will be automatically registered via the factory (requires `@register_handler` decorator or manual registration in `factory.py`).

Example for a new dataset "my_corpus":

```python
from dendritic.dataset_handlers.TextCorpusHandler import TextCorpusHandler
from dendritic.dataset_handlers.factory import register_handler

@register_handler("my_corpus")
class MyCorpusHandler(TextCorpusHandler):
    dataset_name = "username/my_corpus"
    text_column = "text"
```

The `load_default_data` method returns a dictionary with keys 'train' and 'test' (regular Datasets). The evaluation split ratio can be controlled via the `test_size` parameter; set `test_size=0.0` to skip evaluation split.

The handler can then be used in experiments via `--dataset my_corpus`. The mandatory `max_samples` parameter ensures controlled downloads, and streaming is enabled by default.
