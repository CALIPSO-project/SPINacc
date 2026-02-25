import os
import pytest
from utils import compare_nc_files


def _get_forcing_prefix(varlist):
    """Return the forcing file prefix for the configured response format."""
    resp_format = varlist.get("resp", {}).get("format", "regular")
    if resp_format == "unstructured":
        return "forcing_unstructured_"
    elif resp_format == "regular":
        return "forcing_regular_"
    else:
        return f"forcing_{resp_format}_"


def _get_year_range(varlist):
    """Return the year range from the varlist climate section."""
    climate = varlist.get("climate", {})
    return range(climate["year_start"], climate["year_end"] + 1)


def test_compare_forcing_nc_files(reference_path, test_path, varlist):
    """
    Compare forcing netCDF files between reference and test output.

    The file prefix is derived from the response format in varlist:
    - "unstructured" → forcing_unstructured_{year}.nc
    - "regular"      → forcing_regular_{year}.nc
    """
    prefix = _get_forcing_prefix(varlist)
    for year in _get_year_range(varlist):
        compare_nc_files(
            os.path.join(reference_path, f"{prefix}{year}.nc"),
            os.path.join(test_path, f"{prefix}{year}.nc"),
        )
