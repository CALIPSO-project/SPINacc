import os
import pandas as pd
import numpy as np
import pytest


@pytest.mark.skip("Skipped for redundancy of tests.")
def test_compare_all_files(reference_path, test_path):
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
    Compare the old .txt files to the new MLacc_results.csv file
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

    test = reference_results.loc[reference_results["comp"] == "biomass"]

    mlacc_results = pd.read_csv(test_path + "/MLacc_results.csv")

    for comp in comps:
        mlacc_results_comp = mlacc_results.loc[mlacc_results["comp"] == comp]
        reference_results_comp = reference_results.loc[
            reference_results["comp"] == comp
        ]

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
