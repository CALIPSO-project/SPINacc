import netCDF4 as nc
import numpy as np


def compare_npy_files(file1, file2):
    a = np.load(file1, allow_pickle=True)
    b = np.load(file2, allow_pickle=True)

    result = recursive_compare(a, b)

    assert result, f"The contents of {file1} and {file2} are different."


def recursive_compare(arr1, arr2, tol=1e-9):
    # Check if both inputs are ndarrays
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        # Check if both ndarrays have the same shape
        if arr1.shape != arr2.shape:
            return False

        # If they contain objects, iterate and compare recursively
        if arr1.dtype == "object" and arr2.dtype == "object":
            for sub_arr1, sub_arr2 in zip(arr1, arr2):
                if not recursive_compare(sub_arr1, sub_arr2, tol):
                    return False
            return True
        else:
            # Compare the arrays element-wise for floating point numbers
            return np.allclose(arr1, arr2, atol=tol)
    else:
        return False


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
        assert np.allclose(
            values1, values2
        ), f"Variable '{variable}' values are different."

    # Close the netCDF files
    nc1.close()
    nc2.close()
