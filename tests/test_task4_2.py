import pytest
from utils import compare_nc_files


@pytest.mark.parametrize(
    "filename", ["SBG_FGSPIN.340Y.ORC22v8034_22501231_stomate_rest.nc"]
)
def test_compare_nc_files(reference_path, test_path, filename):
    compare_nc_files(reference_path + filename, test_path + filename)
