import numpy as np
import pytest


@pytest.fixture
def reference_path():  # reference EXE_DIR in zenodo
    return "/home/surface10/mrasolon/files_for_zenodo/reference/EXE_DIR/"


@pytest.fixture
def test_path():  # test EXE_DIR
    return "./EXE_DIR/"


@pytest.fixture
def output_file(test_path):  # logging file for reproducibility tests results
    return test_path + "tests_results.txt"
