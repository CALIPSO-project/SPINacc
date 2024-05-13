import netCDF4 as nc
import numpy as np


def compare_npy_files(file1, file2):
    a = np.load(file1, allow_pickle=True)
    b = np.load(file2, allow_pickle=True)
    assert np.all(a == b), f"The contents of {file1} and {file2} are different."


def compare_nc_files(file1_path, file2_path):
    # Open the netCDF files
    nc1 = nc.Dataset(file1_path)
    nc2 = nc.Dataset(file2_path)

    # Compare dimensions
    assert set(nc1.dimensions.keys()) == set(
        nc2.dimensions.keys()
    ), "Dimensions are different."

    # Compare variables
    assert set(nc1.variables.keys()) == set(
        nc2.variables.keys()
    ), "Variables are different."

    # Compare variable values
    for variable in set(nc1.variables.keys()) & set(nc2.variables.keys()):
        values1 = nc1[variable][:]
        values2 = nc2[variable][:]
        assert (
            values1 == values2
        ).all(), f"Variable '{variable}' values are different."

    # Close the netCDF files
    nc1.close()
    nc2.close()
