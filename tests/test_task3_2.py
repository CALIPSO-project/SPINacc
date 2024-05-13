import netCDF4 as nc

# -------------------------------variables----------------------------------------------------
from config import output_file, reference_path, test_path

# reference_path = '/home/surface10/mrasolon/SPINacc_12_7/EXE_DIR_2/'
# test_path = '/home/surface10/mrasolon/SPINacc_12_7/EXE_DIR/'  # Enter the path from your EXE_DIR

# output_file = '/home/surface10/mrasolon/tests/tests_results.txt'  # Define the output file

# ------------------------end of variables----------------------------------------------------


def compare_nc_files(file1_path, file2_path, output_file, print_diff_flag=False):
    # Open the netCDF files
    with open(output_file, "a") as output:
        output.write(" \n")
        output.write("For " + file1_path + " and " + file2_path + "\n")
        output.write(" \n")
        nc1 = nc.Dataset(file1_path)
        nc2 = nc.Dataset(file2_path)

        # Compare dimensions
        if set(nc1.dimensions.keys()) == set(nc2.dimensions.keys()):
            output.write("Dimensions are identical.\n")
        else:
            output.write("Dimensions are different.\n")

        # Compare variables
        if set(nc1.variables.keys()) == set(nc2.variables.keys()):
            output.write("Variables are identical.\n")
        else:
            output.write("Variables are different.\n")

        # Compare variable values
        variable_diff_flag = False  # Flag to track if any variable values are different
        for variable in set(nc1.variables.keys()) & set(nc2.variables.keys()):
            values1 = nc1[variable][:]
            values2 = nc2[variable][:]

            if (values1 == values2).all():
                output.write(f"Variable '{variable}' values are identical.\n")
            else:
                output.write(f"Variable '{variable}' values are different.\n")
                variable_diff_flag = True  # Set the flag to True

                if print_diff_flag:
                    output.write(f"Values in {file1_path}: {values1}\n")
                    output.write(f"Values in {file2_path}: {values2}\n")

        # Display a message if any variable values are different
        if variable_diff_flag:
            output.write("Some variable values are different.\n")

        # Close the netCDF files
        nc1.close()
        nc2.close()


# Example usage
output_file_path = output_file

file1_path = test_path + "SRF_FGSPIN.10Y.ORC22v8034_19101231_sechiba_rest.nc"
file2_path = reference_path + "SRF_FGSPIN.10Y.ORC22v8034_19101231_sechiba_rest.nc"

file3_path = test_path + "OOL_FGSPIN.10Y.ORC22v8034_19101231_driver_rest.nc"
file4_path = reference_path + "OOL_FGSPIN.10Y.ORC22v8034_19101231_driver_rest.nc"

file5_path = test_path + "SBG_FGSPIN.10Y.ORC22v8034_19101231_stomate_rest.nc"
file6_path = reference_path + "SBG_FGSPIN.10Y.ORC22v8034_19101231_stomate_rest.nc"


compare_nc_files(file1_path, file2_path, output_file_path, print_diff_flag=True)
compare_nc_files(file3_path, file4_path, output_file_path, print_diff_flag=True)
compare_nc_files(file5_path, file6_path, output_file_path, print_diff_flag=True)
