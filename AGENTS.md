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
- **Dataset handling**: Uses custom PythonAlpacaHandler with specific prompt formatting requirements

## Critical Gotchas

- **Memory management**: Integration tests include explicit garbage collection (`gc.collect()`) due to large model memory requirements
- **Device placement**: Tests verify CUDA/CPU consistency with detailed debugging output when GPU available
- **Serialization**: Numpy types must be converted to native Python types for JSON serialization in experiment results
- **Test data**: Some edge case tests are marked `@pytest.mark.xfail` due to known limitations

## File Organization

- **Experiments**: All experiment code in `dendritic/experiments/` with specific run patterns
- **Layers**: Custom dendritic layers in `dendritic/layers/` with mathematical correctness verification
- **Handlers**: Dataset handlers in `dendritic/dataset_handlers/` with specific interface requirements
