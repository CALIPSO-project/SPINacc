from utils import recursive_compare
import numpy as np
import xarray
import os
import pytest


@pytest.mark.skip("Skipped as packdata format has changed to netcdf.")
def test_packdata_variable_equivalence(reference_path, test_path):
    """
    Test that the new packdata has at least as many variables as
    the reference
    """
    packdata_ref_file = os.path.join(reference_path, "packdata.npy")
    packdata_ref = np.load(packdata_ref_file, allow_pickle=True).item()

    packdata_test_file = os.path.join(test_path, "packdata.nc")
    packdata_test = xarray.load_dataset(packdata_test_file)
    vars = list(packdata_test.variables.keys())
    vars_ref = dir(packdata_ref)

    # Eliminate all contents of vars_old beginning with "_"
    vars_ref = [var for var in vars_ref if not var.startswith("_")]
    for var in vars_ref:
        assert var in vars, f"Variable {var} is not present in new packdata"

    for var in vars:
        if var in ["lat", "lon"]:
            continue
        print(var)
        assert var in vars_ref, f"Variable {var} is not present in refpackdata"


@pytest.mark.skip("Skipped as packdata format has changed to netcdf.")
def test_packdata_equivalence_npy(reference_path, test_path):
    """
    Test equivalence between the initialised state for DEF_Trunk
    after refactor.

    Note that file has been converted to NetCDF4 and auxil data has now been
    subsumed.
    """

    # Load test data
    packdata_test_file = os.path.join(test_path, "packdata.nc")
    packdata_test = xarray.load_dataset(packdata_test_file)

    # Load reference data
    packdata_ref_file = os.path.join(reference_path, "packdata.npy")
    packdata_ref = np.load(packdata_ref_file, allow_pickle=True).item()

    auxil_ref_file = os.path.join(reference_path, "auxil.npy")
    auxil_ref = np.load(auxil_ref_file, allow_pickle=True).item()

    assert packdata_test.nlat == auxil_ref.nlat
    assert packdata_test.nlon == auxil_ref.nlon
    assert packdata_test.lat_reso == auxil_ref.lat_reso
    assert packdata_test.lon_reso == auxil_ref.lon_reso

    vars = list(packdata_test.variables.keys())
    for v in vars:
        if v == "lat" or v == "lon":
            continue
        else:
            assert recursive_compare(packdata_test[v].values, packdata_ref[v]), (
                f"Variable {v} differs between datasets"
            )

            print(f"Variable {v} matches between datasets.")


def test_packdata_equivalence(reference_path, test_path):
    """
    Test equivalence between the initialised state for DEF_Trunk
    after refactor.

    Note that file has been converted to NetCDF4 and auxil data has now been
    subsumed.
    """

    packdata_test_file = os.path.join(test_path, "packdata.nc")
    packdata_test = xarray.load_dataset(packdata_test_file)

    packdata_ref_file = os.path.join(reference_path, "packdata.nc")
    packdata_ref = xarray.load_dataset(packdata_ref_file)

    assert packdata_test.nlat == packdata_ref.nlat
    assert packdata_test.nlon == packdata_ref.nlon
    assert packdata_test.lat_reso == packdata_ref.lat_reso
    assert packdata_test.lon_reso == packdata_ref.lon_reso

    vars = list(packdata_test.variables.keys())
    for v in vars:
        if v == "lat" or v == "lon":
            continue
        else:
            assert recursive_compare(packdata_test[v].values, packdata_ref[v].values), (
                f"Variable {v} differs between datasets"
            )

            print(f"Variable {v} matches between datasets.")


def test_packdata_variable_equivalence(reference_path, test_path):
    """
    Test that the packdata has at least as many variables as
    the reference
    """
    packdata_test_file = os.path.join(test_path, "packdata.nc")
    packdata_test = xarray.load_dataset(packdata_test_file)

    packdata_ref_file = os.path.join(reference_path, "packdata.nc")
    packdata_ref = xarray.load_dataset(packdata_ref_file)

    vars = list(packdata_test.variables.keys())
    vars_ref = list(packdata_ref.variables.keys())

    # Eliminate all contents of vars_old beginning with "_"
    # vars_ref = [var for var in vars_ref if not var.startswith("_")]
    for var in vars_ref:
        assert var in vars, f"Variable {var} is not present in packdata"

    for var in vars:
        if var in ["lat", "lon"]:
            continue
        print(var)
        assert var in vars_ref, f"Variable {var} is not present in reference packdata"
