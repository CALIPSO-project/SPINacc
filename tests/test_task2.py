# -------------------------------variables----------------------------------------------------
from config import output_file, reference_path, test_path

# reference_path = '/home/surface10/mrasolon/SPINacc_12_7/EXE_DIR_2/'
# test_path = '/home/surface10/mrasolon/SPINacc_12_7/EXE_DIR/'

# output_file = '/home/surface10/mrasolon/tests/tests_results.txt'  # Define the output file
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
# -------------------------End of comparison of dist_all as .txt files--------------------------

# -----------------------Compare packdata and auxil files as binary files-----------------------
file_paths = [
    (reference_path + "IDloc.npy", test_path + "IDloc.npy"),
    (reference_path + "IDsel.npy", test_path + "IDsel.npy"),
    (reference_path + "IDx.npy", test_path + "IDx.npy"),
]
# WRITE TASK
with open(output_file, "a") as out_file:
    out_file.write(" \n")
    out_file.write(
        f"################################### REPRODUCIBILITY TEST FOR TASK 2: ################################### \n"
    )
    out_file.write(" \n")
for file1, file2 in file_paths:
    compare_npy_files(file1, file2, output_file)
# --------------------------end of comparison test code----------------------------------------------------
