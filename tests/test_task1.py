import pytest
from utils import compare_npy_files


@pytest.mark.parametrize(
    "filename",
    [
        "dist_all.npy"
    ],  # "auxil.npy" & "packdata.npy" have been replaced by "packdata.nc"
)
def test_compare_npy_files(reference_path, test_path, filename):
    compare_npy_files(reference_path + filename, test_path + filename)
