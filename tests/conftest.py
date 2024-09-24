import numpy as np
import pytest
import sys
import os


def pytest_addoption(parser):
    # Define a pytest option to pass the config directory
    parser.addoption(
        "--trunk",
        action="store",
        default="DEF_Trunk",
        help="Trunk directory to import modules from",
    )


# Fixture to get the config module from any given Trunk directory
@pytest.fixture(scope="session", autouse=True)
def add_config_dir_to_path(pytestconfig):
    trunk_dir = pytestconfig.getoption("trunk")
    if trunk_dir:
        sys.path.append(trunk_dir)
        print(f"Trunk directory {trunk_dir} added to sys.path")
        try:
            global config  # Make it globally accessible in other parts of the test
            import config
        except ImportError as e:
            raise ImportError(
                f"Failed to import 'my_module' from {trunk_dir}. Error: {e}"
            )
    else:
        raise FileNotFoundError(
            f"Trunk  directory {trunk_dir} does not exist or is not a directory"
        )


@pytest.fixture
def reference_path():
    """Path to reference data."""
    return config.reference_dir


@pytest.fixture
def test_path():
    """Path to test data."""
    return config.results_dir


@pytest.fixture
def output_file(test_path):
    """
    Logging file for reproducibility test results.
    """
    return os.path.join(test_path, "tests_results.txt")
