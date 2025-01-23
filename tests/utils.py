import netCDF4 as nc
import numpy as np


def compare_npy_files(file1, file2):
    """
    Compare the contents of two .npy files.
    """
    file1_npy = np.load(file1, allow_pickle=True)
    file2_npy = np.load(file2, allow_pickle=True)

    assert recursive_compare(file1_npy, file2_npy), (
        f"The contents of {file1} and {file2} are different."
    )


def recursive_compare(arr1, arr2, tol=1e-8):
    """
    Recursively compare the contents of two numpy (nd)arrays.
    """
    # Check if both inputs are ndarrays
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        # Check if both ndarrays have the same shape
        if arr1.shape != arr2.shape:
            print("shapes not the same")
            return False

        # If they contain objects, iterate and compare recursively
        if arr1.dtype == "object" and arr2.dtype == "object":
            for sub_arr1, sub_arr2 in zip(arr1, arr2):
                if not recursive_compare(sub_arr1, sub_arr2, tol):
                    print(sub_arr1, sub_arr2)
                    return False
            return True
        else:
            # Compare the arrays element-wise for floating point numbers
            if np.allclose(arr1, arr2, atol=tol, equal_nan=True) != True:
                print("Values are different!")
                d = ~np.isclose(arr1, arr2, atol=tol, equal_nan=True)
                print(len(d))
                for i, mask in enumerate(d):
                    if False in mask:
                        print(arr1[i][mask], arr2[i][mask])
            # This redundant line evaluates to True
            return np.allclose(arr1, arr2, atol=tol, equal_nan=True)
    else:
        return False


def compare_nc_files(file1_path, file2_path):
    """
    Compare the contents of two netCDF files.
    """

    # Open the netCDF files
    nc1 = nc.Dataset(file1_path)
    nc2 = nc.Dataset(file2_path)

    # Compare dimensions
    assert set(nc1.dimensions.keys()) == set(nc2.dimensions.keys()), (
        "Dimensions are different."
    )

    # Compare variables
    assert set(nc1.variables.keys()) == set(nc2.variables.keys()), (
        "Variables are different."
    )

    # Compare variable values
    for variable in set(nc1.variables.keys()) & set(nc2.variables.keys()):
        values1 = nc1[variable][:]
        values2 = nc2[variable][:]
        assert np.allclose(values1, values2), (
            f"Variable '{variable}' values are different."
        )

    # Close the netCDF files
    nc1.close()
    nc2.close()
