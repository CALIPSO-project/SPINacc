import os

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

    for file in common_files:
        file1_path = os.path.join(reference_path, file)
        file2_path = os.path.join(test_path, file)

        with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:
            content1 = file1.read()
            content2 = file2.read()
            assert (
                content1 == content2
            ), f"File {file} in {reference_path} and {test_path} are different."
