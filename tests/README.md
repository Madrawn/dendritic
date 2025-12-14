# Dendritic Test Suite

This directory contains the test suite for the Dendritic project. The test suite is organized into several categories:

## Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **Edge Case Tests**: Test boundary conditions and unusual inputs
- **Error Handling**: Test error conditions and exception handling

## Test Files

- `test_data_pipelines.py`: Tests for data pipeline functionality
- `test_dataset_handlers.py`: Tests for dataset handling
- `test_edge_cases.py`: Tests for edge cases
- `test_enhancement.py`: Tests for enhancement functionality
- `test_error_handling.py`: Tests for error handling
- `test_integration.py`: Integration tests
- `test_layer.py`: Tests for layer functionality
- `enhance_test.py`: Main enhancement test

## Running Tests

Use the test runner script to execute tests:

```bash
python run_tests.py [options]
```

### Options:
- `--mode`: Specify test mode (unit, integration, edge, all)
- `--parallel`: Run tests in parallel
- `--coverage`: Generate coverage report

### Running specific test categories:
```bash
# Run unit tests
python run_tests.py --mode unit

# Run integration tests with coverage
python run_tests.py --mode integration --coverage
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:
- Common setup/teardown logic
- Mock objects
- Test configuration

## Regression Baseline

The test suite serves as a regression baseline. To validate:
1. Run all tests: `python run_tests.py`
2. Check for any failures
3. Review coverage reports in `htmlcov/` directory

Tests should be deterministic and repeatable. If you encounter flaky tests, mark them with `@pytest.mark.flaky` and investigate.