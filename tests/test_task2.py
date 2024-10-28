import pytest
from utils import compare_npy_files


@pytest.mark.parametrize("filename", ["IDloc.npy", "IDsel.npy", "IDx.npy"])
def test_compare_npy_files(reference_path, test_path, filename):
    """
    Compares the output of the Clustering step to the reference data.

    Note that we only use IDx.npy for step 4. IDSel is only used for plotting.
    """
    compare_npy_files(reference_path + filename, test_path + filename)
