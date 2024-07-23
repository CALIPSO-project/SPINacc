import os

import numpy as np

# -------------------------------variables----------------------------------------------------
from config import output_file, reference_path, test_path

# reference_path = '/home/surface10/mrasolon/SPINacc_12_7/EXE_DIR/'
# test_path = '/home/surface10/mrasolon/SPINacc_12_7/EXE_DIR_3/'  # Enter the path from your EXE_DIR

# output_file = test_path + 'tests_results.txt'  # Define the output file
# ------------------------end of variables----------------------------------------------------


# -------------------------------functions----------------------------------------------------
def compare_npy_files(file1, file2, output_file):
    # Read binary data from the files
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        data1 = f1.read()
        data2 = f2.read()

    # Compare the binary data
    with open(output_file, "a") as out_file:
        if data1 == data2:
            out_file.write(f"The contents of {file1} and {file2} are identical.\n")
        else:
            out_file.write(f"The contents of {file1} and {file2} are different.\n")


# -------------------------------end of functions----------------------------------------------

# ------------------------------ Comparison of dist_all as .txt files--------------------------
# Load the first .npy file
array1 = np.load(reference_path + "dist_all.npy", allow_pickle=True)

# Save the first array as a text file
np.savetxt("dist_all_ref.txt", array1)

# Load the second .npy file
array2 = np.load(test_path + "dist_all.npy", allow_pickle=True)

# Save the second array as a text file
np.savetxt("dist_all_test.txt", array2)

# Load the text files
text_array1 = np.loadtxt("dist_all_ref.txt")
text_array2 = np.loadtxt("dist_all_test.txt")

# Calculate the element-wise absolute error
error_array = np.abs(text_array1 - text_array2)


# Print or use the error array as needed
with open(output_file, "a") as out_file:
    # WRITE TASK
    out_file.write(" \n")
    out_file.write(
        "################################### REPRODUCIBILITY TEST FOR TASK 1 ################################### \n"
    )
    out_file.write(" \n")
    out_file.write("Element-wise absolute error array for dist_all.npy:\n")
    out_file.write(str(error_array) + "\n")
    out_file.write("error_array_size: " + str(error_array.shape) + "\n")

# Delete the text files
os.remove("dist_all_ref.txt")
os.remove("dist_all_test.txt")
# -------------------------End of comparison of dist_all as .txt files--------------------------

# -----------------------Compare packdata and auxil files as binary files-----------------------
file_paths = [
    (reference_path + "auxil.npy", test_path + "auxil.npy"),
    (reference_path + "packdata.npy", test_path + "packdata.npy"),
    (reference_path + "dist_all.npy", test_path + "dist_all.npy"),
]

for file1, file2 in file_paths:
    compare_npy_files(file1, file2, output_file)
# --------------------------end of comparison test code----------------------------------------------------
