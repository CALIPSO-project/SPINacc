import pytest
from utils import compare_nc_files


@pytest.fixture
def file_prefix():
    return "forcing_aligned_"


@pytest.mark.parametrize("year", range(1901, 1911))
def test_compare_nc_files(reference_path, test_path, file_prefix, year):
    file_path_nc1 = f"{reference_path}{file_prefix}{year}.nc"
    file_path_nc2 = f"{test_path}{file_prefix}{year}.nc"
    compare_nc_files(file_path_nc1, file_path_nc2)
