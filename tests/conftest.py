import pytest
import sys
import os
import logging
import os
import subprocess


def setup_windows_compiler():
    if os.name != "nt":
        return

    # Try multiple Visual Studio versions
    vs_versions = ["2022", "2019", "2017"]
    vs_variants = ["Community", "Professional", "Enterprise", "BuildTools"]

    vcvars_path = None
    for version in vs_versions:
        base_path = rf"C:\Program Files\Microsoft Visual Studio\{version}"
        for variant in vs_variants:
            path = os.path.join(base_path, variant, r"VC\Auxiliary\Build\vcvars64.bat")
            if os.path.exists(path):
                vcvars_path = path
                print(f"Found vcvars at {vcvars_path}")
                break
        if vcvars_path:
            break

    if not vcvars_path:
        print("No Visual Studio vcvars64.bat found.")
        return

    # Disable Conda AutoRun by setting environment variable
    env = os.environ.copy()
    env["CONDA_AUTO_RUN"] = "0"

    # Try up to 3 times with a small delay
    for attempt in range(3):
        try:
            # Original command that worked before
            cmd = f'cmd.exe /d /c "{vcvars_path}" && set'
            output = subprocess.check_output(
                cmd, shell=False, text=True, stderr=subprocess.STDOUT, env=env
            )

            for line in output.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
            print("Successfully loaded MSVC environment (bypassing Conda AutoRun).")
            return
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed: {e.output}")
            if attempt < 2:
                import time

                time.sleep(1)
            else:
                print("Failed to load MSVC after 3 attempts.")
                # Fallback: try to locate cl.exe manually
                import glob

                cl_path = None
                for version in vs_versions:
                    for variant in vs_variants:
                        pattern = rf"C:\Program Files\Microsoft Visual Studio\{version}\{variant}\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
                        matches = glob.glob(pattern)
                        if matches:
                            cl_path = matches[0]
                            break
                    if cl_path:
                        break
                if cl_path:
                    cl_dir = os.path.dirname(cl_path)
                    os.environ["PATH"] = cl_dir + ";" + os.environ.get("PATH", "")
                    print(f"Added cl.exe directory to PATH: {cl_dir}")
                else:
                    print("Could not locate cl.exe.")


setup_windows_compiler()
import torch

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

    # Define your custom timeout for integration tests
    INTEGRATION_TIMEOUT = 120

    for item in items:
        # 1. NEW: Check if 'integration' is in the markers
        # We check own_markers to see if it was explicitly tagged
        is_integration = any(
            marker.name == "integration" for marker in item.own_markers
        )

        if is_integration:
            # Add the timeout marker dynamically
            item.add_marker(pytest.mark.timeout(INTEGRATION_TIMEOUT))
        if not item.own_markers:
            unmarked_tests.append(item.nodeid)

    if unmarked_tests:
        msg = "The following tests are missing pytest markers:\n" + "\n".join(
            unmarked_tests
        )
        pytest.exit(msg)
