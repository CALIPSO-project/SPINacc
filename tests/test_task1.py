import pytest
from utils import compare_npy_files
import os


@pytest.mark.parametrize(
    "filename",
    [
        "dist_all.npy"
    ],  # "auxil.npy" & "packdata.npy" have been replaced by "packdata.nc"
)
def test_compare_npy_files(reference_path, test_path, filename):
    """
    Compares the output of the clustering_test function to the reference data.

    See test_tast2.py for a more robust test of the clustering step.
    """
    compare_npy_files(
        os.path.join(reference_path, filename), os.path.join(test_path, filename)
    )
