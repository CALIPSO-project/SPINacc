import pytest
import numpy as np
from tests.conftest import reference_path, test_path, output_file, compare_npy_files


def test_dist_data():
    array1 = np.load(reference_path + 'dist_all.npy', allow_pickle=True)
    array2 = np.load(test_path + 'dist_all.npy', allow_pickle=True)
    
    assert np.allclose(array1, array2, atol=1e-6), "the two arrays are different"

    # # Calculate the element-wise absolute error
    # error_array = np.abs(array1 - array2)
    
    # # Print or use the error array as needed
    # with open(output_file, 'a') as out_file:
    #     # WRITE TASK 
    #     out_file.write(" \n")
    #     out_file.write(f"################################### REPRODUCIBILITY TEST FOR TASK 1 ################################### \n")
    #     out_file.write(" \n")
    #     out_file.write("Element-wise absolute error array for dist_all.npy:\n")
    #     out_file.write(str(error_array) + '\n')
    #     out_file.write("error_array_size: " + str(error_array.shape) + '\n')


# -----------------------Compare packdata and auxil files as binary files-----------------------
@pytest.mark.parametrize(
    "path_pair",
    [
        (reference_path + 'auxil.npy', test_path + 'auxil.npy'),
        (reference_path + 'packdata.npy', test_path + 'packdata.npy'),
        (reference_path + 'dist_all.npy', test_path + 'dist_all.npy')
    ]
)
def test_compare_npy_files(path_pair):
    compare_npy_files(*path_pair, output_file)
