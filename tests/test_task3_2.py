import pytest
from utils import compare_nc_files


@pytest.mark.parametrize(
    "filename",
    [
        "SRF_FGSPIN.10Y.ORC22v8034_19101231_sechiba_rest.nc",
        "OOL_FGSPIN.10Y.ORC22v8034_19101231_driver_rest.nc",
        "SBG_FGSPIN.10Y.ORC22v8034_19101231_stomate_rest.nc",
    ],
)
def test_compare_nc_files(reference_path, test_path, filename):
    compare_nc_files(reference_path + filename, test_path + filename)
