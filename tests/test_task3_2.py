import os
import pytest
from utils import compare_nc_files


def _get_restart_filenames(varlist):
    """
    Return output restart file names based on the configured response format.

    For "unstructured" format, the output files have an "_unstructured" suffix
    (e.g. ``my_restart_unstructured.nc``), matching the output of forcing.write().
    For other formats the basename is unchanged.
    """
    resp_format = varlist.get("resp", {}).get("format", "regular")
    restart_paths = varlist.get("restart", [])
    filenames = []
    for path in restart_paths:
        basename = os.path.basename(path)
        if resp_format == "unstructured":
            filenames.append(basename.replace(".nc", "_unstructured.nc"))
        else:
            filenames.append(basename)
    return filenames


def test_compare_restart_nc_files(reference_path, test_path, varlist):
    """
    Compare restart netCDF files between reference and test output.

    File names are derived from varlist based on the response format:
    - "unstructured" → {original_name}_unstructured.nc
    - other formats  → {original_name} (unchanged)
    """
    filenames = _get_restart_filenames(varlist)
    for filename in filenames:
        compare_nc_files(
            os.path.join(reference_path, filename),
            os.path.join(test_path, filename),
        )
