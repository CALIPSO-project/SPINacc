import os
import pandas as pd
import numpy as np
import pytest


@pytest.mark.skip("Skipped for redundancy of tests.")
def test_compare_all_files(reference_path, test_path):
    """
    This function compares all files in the reference_path to the test_path.

    This will compare .txt files against each other. No longer relevant
    as we have transitioned to .csv files for MLacc_results.
    """
    files1 = [
        f
        for f in os.listdir(reference_path)
        if os.path.isfile(os.path.join(reference_path, f))
    ]
    files2 = [
        f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))
    ]

    common_files = set(files1) & set(files2)

    print(common_files)

    for file in common_files:
        file1_path = os.path.join(reference_path, file)
        file2_path = os.path.join(test_path, file)

        with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:
            content1 = file1.read()
            content2 = file2.read()
            assert (
                content1 == content2
            ), f"File {file} in {reference_path} and {test_path} are different."


def get_df_comp(EXE_DIR, file_name, comp):
    """
    Read the component i.e. 'som', 'litter' or 'biomass' .txt files and return a DataFrame

    Consider preprocessing all of the .txt files into a single DataFrame before comparing.
    """

    file_path = os.path.join(EXE_DIR, file_name)

    # Determine the column name based on the file name
    column_name = file_name.split(".")[0].split("_")[1:]
    column_name = "_".join(column_name)

    # Read the file and remove the trailing comma from each line
    with open(file_path, "r") as file:
        data = [line.strip().strip(",").split(",") for line in file]

    data = [[float(value) for value in row] for row in data]

    reshaped_data = []

    for index, row in enumerate(data, start=1):
        for col_num, value in enumerate(row, start=1):
            reshaped_data.append([comp, index, col_num, value])

    df = pd.DataFrame(reshaped_data, columns=["comp", "var", "var2", column_name])

    return df


def test_compare_csv_to_txt(reference_path, test_path):
    """
    Compare the old .txt files to the new MLacc_results.csv file.

    We construct a new DataFrame with the contents of the .txt files and compare.
    This currently only works for the 'som' component.
    """
    comps = ["som", "biomass", "litter"]

    metrics = [
        "R2",
        "dNRMSE",
        "slope",
        "sNRMSE",
        "iNRMSE",
        "f_SDSD",
        "f_SB",
        "f_LSC",
    ]

    # construct a new dataframe with the contents of the .txt files.
    reference_results = pd.DataFrame()

    for comp in comps:
        reference_comp = pd.DataFrame()
        for metric in metrics:
            df = get_df_comp(reference_path, comp + "_" + metric + ".txt", comp)
            reference_comp = pd.concat([reference_comp, df], axis=1)
            reference_comp = reference_comp.loc[:, ~reference_comp.columns.duplicated()]

        reference_results = pd.concat(
            [reference_results, reference_comp], ignore_index=True, axis=0
        )

    mlacc_results = pd.read_csv(test_path + "/MLacc_results.csv")
    mlacc_results = mlacc_results.sort_values(
        by=["comp", "ipft", "ivar"], ignore_index=True
    )

    for comp in comps:
        mlacc_results_comp = mlacc_results.loc[mlacc_results["comp"] == comp]

        reference_results_comp = reference_results.loc[
            reference_results["comp"] == comp
        ]

        if comp == "biomass":
            mlacc_results_comp = mlacc_results_comp.sort_values(
                by=["ivar", "ipft"], ignore_index=True
            )

        if comp == "litter":
            df_ab = mlacc_results_comp[mlacc_results_comp["var"].str.endswith("_ab")]
            df_be = mlacc_results_comp[mlacc_results_comp["var"].str.endswith("_be")]
            mlacc_results_comp = pd.concat([df_ab, df_be], axis=0, ignore_index=True)

        for metric in metrics:
            print("metrics ", metric, comp)
            comparison = np.isclose(
                mlacc_results_comp[metric],
                reference_results_comp[metric],
                atol=1e-2,
                equal_nan=True,
            )
            if comparison.all():
                print(f"All values in {metric} match MLacc_results.csv")
            else:
                print(f"Some values in {metric} do not match MLacc_results.csv")
                print(
                    pd.concat(
                        [
                            mlacc_results_comp[~comparison][["comp", "var", metric]],
                            reference_results_comp[~comparison][
                                ["comp", "var", metric]
                            ],
                        ],
                        axis=1,
                    )
                )
                assert False
    assert True
