from utils import recursive_compare
import numpy as np
import xarray


def test_packdata_equivalence(reference_path, test_path):
    """
    Test equivalence between the initialised state for DEF_Trunk
    after refactor.

    Note that file changed to NetCDF4 file and auxil data has been
    subsumed.
    """

    packdata_ref_file = reference_path + "packdata.npy"
    packdata_test_file = test_path + "packdata.nc"

    packdata_ref = np.load(packdata_ref_file, allow_pickle=True).item()
    packdata_test = xarray.load_dataset(packdata_test_file)

    vars = list(packdata_test.variables.keys())
    for v in vars:
        if v == "lat" or v == "lon":
            continue
        else:
            assert recursive_compare(
                packdata_test[v].values, packdata_ref[v]
            ), f"Variable {v} differs between datasets"
