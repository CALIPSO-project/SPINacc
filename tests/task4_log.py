import os

# -------------------------------variables----------------------------------------------------
from config import output_file, reference_path, test_path

# reference_path = '/home/surface10/mrasolon/SPINacc_12_7/EXE_DIR_2/'
# test_path = '/home/surface10/mrasolon/SPINacc_12_7/EXE_DIR/'  # Enter the path from your EXE_DIR

# output_file = '/home/surface10/mrasolon/tests/tests_results.txt'  # Define the output file
# ------------------------end of variables----------------------------------------------------


def compare_files(dir1, dir2, output_file):
    files1 = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
    files2 = [f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

    common_files = set(files1) & set(files2)

    with open(output_file, "a") as output:
        output.write(" \n")
        output.write(
            "################################### REPRODUCIBILITY TEST FOR TASK 4 ################################### \n"
        )
        output.write(" \n")
        for file in common_files:
            file1_path = os.path.join(dir1, file)
            file2_path = os.path.join(dir2, file)

            with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:
                content1 = file1.read()
                content2 = file2.read()

                if content1 == content2:
                    output.write(f"File {file} in {dir1} and {dir2} is identical.\n")
                else:
                    output.write(f"File {file} in {dir1} and {dir2} is different.\n")


if __name__ == "__main__":
    directory1 = reference_path
    directory2 = test_path
    output_file_path = output_file

    compare_files(directory1, directory2, output_file_path)
