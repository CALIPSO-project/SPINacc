import netCDF4 as nc

# -------------------------------variables----------------------------------------------------
from config import output_file, reference_path, test_path

# Define the base file path
# reference_path='/home/surface10/mrasolon/SPINacc_12_7/EXE_DIR_2/'
# test_path='/home/surface10/mrasolon/SPINacc_12_7/EXE_DIR/'
base_file_path_1 = reference_path + "forcing_aligned_"
base_file_path_2 = test_path + "forcing_aligned_"
# List of years to compare (assuming from 1902 to 1910)
years_to_compare = range(1901, 1911)

# Open a file for writing
# output_file_path = '/home/surface10/mrasolon/tests/tests_results.txt'
output_file_path = output_file
# -----------------------end of variables------------------------------------------------------

with open(output_file_path, "a") as output_file:
    output_file.write(" \n")
    output_file.write(
        "################################### REPRODUCIBILITY TEST FOR TASK 3 ################################### \n"
    )
    output_file.write(" \n")
    # Loop through each year
    for year in years_to_compare:
        # Construct the file paths for the current year
        file_path_nc1 = f"{base_file_path_1}{year}.nc"
        file_path_nc2 = f"{base_file_path_2}{year}.nc"

        # Open the NetCDF files
        dataset_nc1 = nc.Dataset(file_path_nc1)
        dataset_nc2 = nc.Dataset(file_path_nc2)

        # Compare dimensions
        if set(dataset_nc1.dimensions.keys()) == set(dataset_nc2.dimensions.keys()):
            output_file.write(f"DIMENSIONS are identical in both files for {year}.\n")
        else:
            output_file.write(f"DIMENSIONS differ between the two files for {year}.\n")

        # Compare variables
        variables_nc1 = set(dataset_nc1.variables.keys())
        variables_nc2 = set(dataset_nc2.variables.keys())

        if variables_nc1 == variables_nc2:
            output_file.write(f"VARIABLES are identical in both files for {year}.\n")
        else:
            output_file.write(f"VARIABLES differ between the two files for {year}.\n")

        # List of variables to compare values
        variables_to_compare = ["Tair", "PSurf", "Qair", "Rainf", "Snowf"]

        # Iterate through variables and compare values
        for var_name in variables_to_compare:
            # Get variable from each file
            variable_nc1 = dataset_nc1.variables[var_name][:]
            variable_nc2 = dataset_nc2.variables[var_name][:]

            # Compare the values
            if (variable_nc1 == variable_nc2).all():
                output_file.write(
                    f"{var_name} values are identical in both files for {year}.\n"
                )
            else:
                output_file.write(
                    f"{var_name} values differ between the two files for {year}.\n"
                )

        # Close the NetCDF files for the current year
        dataset_nc1.close()
        dataset_nc2.close()
