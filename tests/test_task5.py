import numpy as np
import json
import sys
import os
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
tools_path = os.path.join(current_dir, "../Tools/")
sys.path.insert(0, tools_path)  # Use insert(0, ...) to prioritize this path
tools_path = os.path.join(current_dir, "../")
sys.path.insert(0, tools_path)  # Use insert(0, ...) to prioritize this path

print(tools_path)
import eval_plot_un


def test_visualisation():
    resultpath = Path("tests/data/")

    with open(resultpath / "varlist.json", "r") as f:
        varlist = json.loads(f.read())

    Yvar = varlist["resp"]["variables"]
    for ipool in Yvar.keys():
        # if ipool!="litter":continue
        subpool_name = varlist["resp"]["pool_name_" + ipool]
        npfts = varlist["resp"]["npfts"]
        subLabel = varlist["resp"]["sub_item"]
        pp = varlist["resp"]["dim"][ipool]
        sect_n = varlist["resp"]["sect_n"][ipool]
        if pp[0] == "pft":
            dims = np.array([0, 1])
        else:
            dims = np.array([1, 0])
        eval_plot_un.plot_metric(
            resultpath, npfts, ipool, subLabel, dims, sect_n, subpool_name
        )

    assert True
