
This file provides guidance to AI Agents when working with code in this repository. Please follow where possible.

## Project Overview

**Dendritic** is a Python machine learning research project implementing dendritic neural network layers for polynomial feature enhancement. The core innovation is adding learnable polynomial (quadratic) features to linear transformations via asymmetric formulation `W₁x ⊙ W₂x`, which optimizes better than symmetric `(Px)²`.

## Development Commands

### Important Notes
- **Running shell commands**: Commands like `python -c` that are followed by multiline strings, will output nothing. Either write a proper .py script, or rather add a proper unit test.

### Testing
- **Run all tests**: `python run_tests.py` (custom test runner with mode selection)
- **Run specific test types**: `python run_tests.py --mode unit|integration|edge|all`
- **Run tests in parallel**: `python run_tests.py --parallel`
- **Run with coverage**: `python run_tests.py --coverage`
- **Run specific test by keyword**: `python run_tests.py -k "test_forward_pass"`

### Coverage Requirements
- Minimum 80% coverage threshold (configured in `.coveragerc`)
- Coverage reports generated to `htmlcov/` directory
- Branch coverage enabled

### Dependencies
- Install: `pip install -e .` (editable install from `pyproject.toml`)
- Core dependencies: `torch`, `transformers`, `datasets`, `tqdm`
- Python: `>=3.7`

## Architecture

### Core Components
1. **Dendritic Layers** (`dendritic/layers/`): Custom neural network layers implementing polynomial feature enhancement
   - `DendriticLayer`: Core asymmetric formulation `W₁x ⊙ W₂x`
   - `DendriticMLP`: MLP with dendritic layers
   - `DendriticStack`: Stacked dendritic layers
   - `InstabilityGatedAttention`: Experimental attention mechanism

2. **Enhancement System** (`dendritic/enhancement.py`): Converts standard `nn.Linear` layers to dendritic layers
   - `enhance_model_with_dendritic()`: Main enhancement function
   - Polynomial rank: `poly_rank="auto"` computes as `max(4, input_dim // 64)`

3. **Experiment Framework** (`dendritic/experiments/`): Two-phase experiments (pretraining + finetuning) plus specialized experiments
   - `run_experiments.py`: Main experiment runner supporting pretraining, finetuning, and confidence experiments
   - Config-driven with dataclass-based configs (`PretrainingConfig`, `FinetuningConfig`, `ConfidenceExperimentConfig`)
   - Multi-seed experiments (default 5 seeds) for statistical significance
   - **Confidence-aware experiments**: Compare ConfidenceAwareGPT (two-pass lookahead) vs standard MiniGPT

4. **Dataset Handlers** (`dendritic/dataset_handlers/`): Factory pattern for dataset loading
   - `BaseDatasetHandler`: Abstract base class
   - `TextCorpusHandler`: For text datasets (WikiText, OpenWebMath, TinyStories, etc.)
   - `PythonAlpacaHandler`: For instruction datasets with prompt formatting
   - `TinyStoriesHandler`: For the TinyStories dataset (simple children's stories)
   - Mandatory `max_samples` parameter (no unbounded downloads)
   - Default `streaming=True` for large datasets

### Key Architectural Patterns

#### Model Variants
The system supports four model variants that must be compared statistically:
- **baseline**: Standard transformer with `nn.Linear` layers
- **dendritic**: Transformer with dendritic-enhanced MLP blocks
- **stack**: Transformer with stacked dendritic layers
- **baseline_wave**: Baseline with sinusoidal activations

#### Experiment Flow
1. **Pretraining**: Language modeling on text corpora
2. **Finetuning**: Instruction tuning on formatted datasets
3. **Analysis**: Statistical comparison between model variants
4. **Visualization**: Built-in result visualization utilities

#### Data Flow
```
Dataset Handler → Preprocessing → Model (baseline/dendritic/stack) → Results → Analysis
```

## Critical Implementation Details

### CUDA Requirement
- **All tests assume CUDA availability**: `assert torch.cuda.is_available()` in `run_tests.py:6`
- **Device placement**: Tests verify CUDA/CPU consistency with detailed debugging output
- **Memory management**: Integration tests include explicit `gc.collect()` due to large model memory requirements

### Test Organization
- **Strict marker system**: All tests must use `@pytest.mark.unit|integration|edge`
- **Marker enforcement**: Configured in `tests/conftest.py`
- **Parallel execution**: Supported via `--parallel` flag
- **Timeout**: 45-second timeout per test (configured in `pyproject.toml`)

### Serialization Requirements
- **Numpy to Python**: Must convert numpy types to native Python types for JSON serialization
- **Result storage**: JSON format with comprehensive metadata
- **Model checkpoints**: `.pth` and `.pt` files

### Dataset Handling Constraints
- **Mandatory limits**: `max_samples` required for all dataset loads (no default)
- **In-memory splitting**: All datasets split via `Dataset.train_test_split`
- **Evaluation split**: Default `test_size=0.1`, can be configured via handler parameter
- **Factory registration**: New datasets registered via `@register_handler` decorator

## Adding New Components

### New Dataset Handler
1. Create handler class inheriting from `TextCorpusHandler` or `BaseDatasetHandler`
2. Set class attributes `dataset_name` and `text_column`
3. Optionally override `load_default_data` for special configuration
4. Register with `@register_handler("dataset_name")` decorator
5. Handler will be available via `--dataset dataset_name` in experiments

### New Model Variant
1. Implement model following existing patterns in `dendritic/experiments/models/`
2. Ensure parameter matching with baseline model
3. Add to model comparison logic in experiment analysis
4. Include comprehensive tests covering forward/backward passes

### New Dendritic Layer
1. Implement in `dendritic/layers/` following existing layer patterns
2. Include mathematical correctness verification
3. Add unit tests for forward/backward passes
4. Update `layer.py` imports and documentation

## Common Pitfalls

1. **Memory leaks**: Always include explicit `gc.collect()` in integration tests
2. **Serialization errors**: Convert numpy types before JSON serialization
3. **Test marker violations**: All tests must have proper pytest markers
4. **Dataset limits**: Never load datasets without `max_samples` parameter
5. **CUDA assumptions**: Tests will fail without GPU availability
6. **Polynomial rank**: Understand `poly_rank="auto"` computes as `max(4, input_dim // 64)`

## Reference Files

- **Build/Test**: `pyproject.toml`, `run_tests.py`, `.coveragerc`
- **Core Logic**: `dendritic/enhancement.py`, `dendritic/layers/layer.py`
- **Experiments**: `dendritic/experiments/run_experiments.py`
- **Configuration**: `dendritic/experiments/utils/PretrainingConfig.py`
- **Testing**: `tests/conftest.py`, `tests/README.md`
- **Agent Guidance**: `AGENTS.md`, `.roo/rules-code/AGENTS.md`, `.roo/rules-architect/AGENTS.md`