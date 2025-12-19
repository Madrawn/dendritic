import pytest
import sys
import os
import logging

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)

# Register markers
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark as unit test")
    config.addinivalue_line("markers", "integration: mark as integration test")
    config.addinivalue_line("markers", "edge: mark as edge case test")


def pytest_collection_modifyitems(config, items):
    """
    Hook to check if tests have markers.
    Fails the test run if any test is missing a marker.
    """
    # If you only want this check to run with a specific flag (optional),
    # you can wrap this block in an `if config.getoption(...)` check.
    
    unmarked_tests = []
    
    for item in items:
        # Check if the test item has any own markers (added via @pytest.mark...)
        # Note: We check `own_markers` to see markers explicitly applied to the test,
        # rather than inherited ones (depending on your strictness requirements).
        if not item.own_markers:
            unmarked_tests.append(item.nodeid)
            
    if unmarked_tests:
        # Create a readable error message
        msg = "The following tests are missing pytest markers:\n" + "\n".join(unmarked_tests)
        
        # You can choose to just print a warning:
        # print(f"\nWARNING: {msg}")
        
        # OR force a hard failure so CI fails:
        pytest.exit(msg)