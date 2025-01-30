import pytest
from utils import compare_nc_files
import os


@pytest.mark.parametrize(
    "filename", ["SBG_FGSPIN.340Y.ORC22v8034_22501231_stomate_rest.nc"]
)
def test_compare_nc_files(reference_path, test_path, filename):
    """
    Compares restart file from reference to test

    This test is only for the stomate restart file.

    Args:
        reference_path (str): Path to the reference file.
        test_path (str): Path to the test file.
        filename (str): Filename of the file to compare.

    """
    compare_nc_files(
        os.path.join(reference_path, filename), os.path.join(test_path, filename)
    )
