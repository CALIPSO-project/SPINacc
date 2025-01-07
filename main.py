#!/usr/bin/env python3
##### -*- coding: utf-8 -*-
"""
MLacc - Machine-Learning-based acceleration of spin-up

This script orchestrates the entire MLacc workflow, including:
- Data preparation and clustering
- Machine learning model training and prediction
- Result evaluation and visualization

The workflow is controlled by configuration settings and can be run in different modes
depending on the specified tasks.

Copyright Laboratoire des Sciences du Climat et de l'Environnement (LSCE)
          Unite mixte CEA-CNRS-UVSQ

Code manager:
Daniel Goll, LSCE, dsgoll123@gmail.com

This software is developed by Yan Sun, Yilong Wang and Daniel Goll.......

This software is governed by the XXX license
XXXX <License content>
"""

from Tools import *

import numpy as np
import xarray
import subprocess

# Print Python version
print(sys.version)

#
# Read configuration file
#

if len(sys.argv) < 2:
    dir_def = "DEF_Trunk/"
else:
    dir_def = sys.argv[1]

sys.path.append(dir_def)
import config

# Define task
itask = str(config.tasks)

# Define result directory
resultpath = config.results_dir

# Create results directory if it does not exist
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

logfile = open(config.logfile, "w", buffering=1)
check.display("Running task: %s" % itask, logfile)
check.display("Results are stored at: " + resultpath, logfile)

# Write the configuration to the log file in the results directory
with open(dir_def + "config.py", "r") as f:
    check.display(f.read(), logfile)

check.display("DEF directory: " + dir_def, logfile)

# Read list of variables
with open(dir_def + "varlist.json", "r") as f:
    varlist = json.loads(f.read())

# Load stored results or start from scratch
if not config.start_from_scratch:
    check.display("Read from previous results...", logfile)
    packdata = xarray.load_dataset(resultpath + "packdata.nc")
else:
    check.display("MLacc start from scratch...", logfile)
    # Initialize packaged data
    packdata = readvar(varlist, config, logfile)
    if os.path.exists(resultpath + "packdata.nc"):
        refdata = xarray.load_dataset(resultpath + "packdata.nc")
        assert (refdata == packdata).all()
    packdata.to_netcdf(resultpath + "packdata.nc")

# Define random seed
iseed = config.random_seed
random.seed(config.random_seed)
np.random.seed(iseed)
check.display("Random seed = %i" % iseed, logfile)

# Leave-one-out cross validation
loocv = config.leave_one_out_cv

# Check if parallel exists in config
if hasattr(config, "parallel"):
    parallel = config.parallel
else:
    parallel = True

# Task 1: Test clustering (optional)
if "1" in itask:
    dis_all = Cluster.Cluster_test(packdata, varlist, logfile)
    # added line
    np.random.seed(iseed)
    dis_all.dump(resultpath + "dist_all.npy")
    check.display(
        "Test clustering done!\nResults have been stored as dist_all.npy", logfile
    )

    # Plot clustering results
    fig, ax = plt.subplots()
    lns = []
    for ipft in range(dis_all.shape[1]):
        lns += ax.plot(packdata.Ks, dis_all[:, ipft])
    plt.legend(lns, varlist["clustering"]["pfts"], title="PFT")
    ax.set_ylabel(
        "Sum of squared distances of samples to\ntheir closest cluster center"
    )
    ax.set_xlabel("K-value (cluster size)")
    fig.savefig(resultpath + "dist_all.png")
    plt.close("all")
    check.display(
        "Test clustering results plotted!\nResults have been stored as dist_all.png",
        logfile,
    )
    # Run test of reproducibility for the task if yes
    if config.repro_test_task_1:
        subprocess.run(
            ["python", "-m", "pytest", "tests/test_task1.py", "--trunk", dir_def]
        )
        check.display(
            "Task 1 reproducibility test results have been stored in tests_results.txt",
            logfile,
        )
    check.display("Task 1: done", logfile)

# Task 2: Clustering
if "2" in itask:
    random.seed(config.random_seed)
    K = config.kmeans_clusters
    check.display("Kmean algorithm, K=%i" % K, logfile)
    IDx, IDloc, IDsel = Cluster.Cluster_all(
        packdata, varlist, K, logfile, config.take_unique
    )
    np.savetxt(resultpath + "IDx.txt", IDx, fmt="%.2f")
    IDx.dump(resultpath + "IDx.npy")
    IDloc.dump(resultpath + "IDloc.npy")
    IDsel.dump(resultpath + "IDsel.npy")
    check.display("Clustering done!\nResults have been stored as IDx.npy", logfile)

    # Plot clustering results
    kpfts = varlist["clustering"]["pfts"]
    for ipft in range(len(kpfts)):
        fig, ax = plt.subplots()
        m = Basemap()
        m.drawcoastlines()
        m.scatter(IDloc[ipft][:, 1], IDloc[ipft][:, 0], s=10, marker="o", c="gray")
        m.scatter(IDsel[ipft][:, 1], IDsel[ipft][:, 0], s=10, marker="o", c="red")
        fig.savefig(resultpath + "ClustRes_PFT%i.png" % kpfts[ipft])
        plt.close("all")
    check.display(
        "Clustering results plotted!\nResults have been stored as ClustRes_PFT*.png",
        logfile,
    )
    # Run reproducibility tests for task 2
    if config.repro_test_task_2:
        subprocess.run(
            ["python", "-m", "pytest", "tests/test_task2.py", "--trunk", dir_def]
        )
        check.display(
            "Task 2 reproducibility test results have been stored in tests_results.txt",
            logfile,
        )
    check.display("Task 2: done", logfile)

# Task 3: Build aligned forcing and aligned restart files (optional)
if "3" in itask:
    check.check_file(resultpath + "IDx.npy", logfile)
    IDx = np.load(resultpath + "IDx.npy", allow_pickle=True)
    forcing.write(varlist, resultpath, IDx)
    # Run test of reproducibility for task 3
    if config.repro_test_task_3:
        subprocess.run(
            ["python", "-m", "pytest", "tests/test_task3.py", "--trunk", dir_def]
        )
        subprocess.run(
            ["python", "-m", "pytest", "tests/test_task3_2.py", "--trunk", dir_def]
        )
        check.display(
            "Task 3 reproducibility test results have been stored in tests_results.txt",
            logfile,
        )
    check.display("Task 3: done", logfile)

# Task 4: Machine learning and extrapolation
if "4" in itask:
    random.seed(config.random_seed)
    np.random.seed(iseed)

    # All predictor variable names (X)
    var_pred_name1 = varlist["pred"]["allname"]

    # LAI and NPP predictor variable names (X)
    var_pred_name2 = varlist["pred"]["allname_pft"]

    # All feature names (X)
    var_pred_name = var_pred_name1 + var_pred_name2

    # Response variable names (Y)
    Yvar = varlist["resp"]["variables"]

    check.check_file(resultpath + "IDx.npy", logfile)
    IDx = np.load(resultpath + "IDx.npy", allow_pickle=True)

    packdata.attrs.update(
        Nv_nopft=len(var_pred_name1),
        Nv_total=len(var_pred_name),
        var_pred_name=var_pred_name,
        Nlat=np.trunc((90 - IDx[:, 0]) / packdata.lat_reso).astype(int),
        Nlon=np.trunc((180 + IDx[:, 1]) / packdata.lon_reso).astype(int),
    )
    labx = ["Y"] + list(packdata.data_vars) + ["pft"]
    # labx = ["Y"] + var_pred_name + ["pft"]
    # Copy the restart file to be modified
    targetfile = (
        varlist["resp"]["targetfile"]
        if "targetfile" in varlist["resp"]
        else varlist["resp"]["sourcefile"]
    )
    restfile = resultpath + targetfile.split("/")[-1]
    os.system("cp -f %s %s" % (targetfile, restfile))
    # Add rights to manipulate file:
    os.chmod(restfile, 0o644)

    for alg in config.algorithms:
        result = []
        for ipool in Yvar.keys():
            check.display("processing %s..." % ipool, logfile)
            res_df = ML.MLloop(
                packdata,
                ipool,
                logfile,
                varlist,
                labx,
                config,
                restfile,
                alg,
                parallel,
                seed=iseed,
            )
            result.append(res_df)
            # Debugging
            # break

        res_df = pd.concat(result, keys=Yvar.keys(), names=["comp"])
        scores = res_df.mean()[["R2", "slope"]].to_frame().T
        scores = scores.assign(alg=alg).set_index("alg")
        path = Path(resultpath + "ML_log.csv")
        scores.to_csv(path, mode="a", header=not path.exists())

        res_path = Path(resultpath + "MLacc_results.csv")
        # if res_path.exists():
        #     ref_df = pd.read_csv(res_path, index_col=[0, 1, 2])
        #     perf_diff = res_df["slope"] - ref_df["slope"]
        #     if perf_diff.mean() > 0 and (perf_diff > 0).mean() > 0.5:
        #         res_df.to_csv(res_path)
        #     else:
        #         print("Degraded performance:", perf_diff.mean(), (perf_diff > 0).mean())
        # else:
        res_df.to_csv(res_path)

    # Additional variables need to be handled in the restart files which are not state variables of ORCHIDEE
    if "additional_vars" not in varlist["resp"]:
        # Handle the case where 'additional_vars' is not present
        print("We only modify true state variables of ORCHIDEE")
    else:
        additional_vars = varlist["resp"]["additional_vars"]

        for var in additional_vars:
            check.display("processing %s..." % var, logfile)
            restnc = Dataset(restfile, "a")
            # All variables derive from npp longterm prediction
            restvar = restnc["npp_longterm"]
            restvar1 = restnc[var]

            if var == "gpp_week" or var == "maxgppweek_lastyear" or var == "gpp_daily":
                tmpvar = restvar[:] * 2.0
            else:
                tmpvar = restvar[:]

            restvar1[:] = tmpvar
            restnc.close()
        # Run reproducibility tests for task 4
    if config.repro_test_task_4:
        subprocess.run(
            ["python", "-m", "pytest", "tests/test_task4.py", "--trunk", dir_def]
        )
        subprocess.run(
            ["python", "-m", "pytest", "tests/test_task4_2.py", "--trunk", dir_def]
        )
        check.display(
            "Task 4 reproducibility test results have been stored in tests_results.txt",
            logfile,
        )
    check.display("Task 4: done", logfile)

# Task 5: Evaluation
if "5" in itask:
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
        if loocv:
            eval_plot_loocv_un.plot_metric(
                resultpath, npfts, ipool, subLabel, dims, sect_n, subpool_name
            )
        else:
            continue
    check.display("Task 5: done", logfile)
