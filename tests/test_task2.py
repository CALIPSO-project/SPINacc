import pytest
from utils import compare_npy_files


@pytest.mark.parametrize("filename", ["IDloc.npy", "IDsel.npy", "IDx.npy"])
def test_compare_npy_files(reference_path, test_path, filename):
    compare_npy_files(reference_path + filename, test_path + filename)
