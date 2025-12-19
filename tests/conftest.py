import pytest
import sys
import os
import logging

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Configure logging
logging.basicConfig(level=logging.INFO)

# 1. NEW: Add a command line option to enable slow tests
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="run benchmark tests"
    )

# Register markers
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark as unit test")
    config.addinivalue_line("markers", "integration: mark as integration test")
    config.addinivalue_line("markers", "edge: mark as edge case test")
    # 2. NEW: Register the slow marker so strict checks don't fail
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    """
    Hook to modify collected items.
    1. Skips slow tests if --runslow is not provided.
    2. Fails the test run if any test is missing a marker.
    """
    
    # --- 3. NEW: Logic to skip slow tests ---
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--benchmark"):
        skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_benchmark)

    # --- EXISTING: Logic to check for missing markers ---
    unmarked_tests = []
    
    for item in items:
        # Check if the test item has any own markers
        if not item.own_markers:
            unmarked_tests.append(item.nodeid)
            
    if unmarked_tests:
        msg = "The following tests are missing pytest markers:\n" + "\n".join(unmarked_tests)
        pytest.exit(msg)