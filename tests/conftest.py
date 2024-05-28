import numpy as np
import pytest


config_path = "DEF_Trunk/MLacc.def"


@pytest.fixture
def reference_path():  # reference EXE_DIR in zenodo
    with open(config_path) as f:
        return f.readlines[25].strip()


@pytest.fixture
def test_path():  # test EXE_DIR
    return "./EXE_DIR/"


@pytest.fixture
def output_file(test_path):  # logging file for reproducibility tests results
    return test_path + "tests_results.txt"
