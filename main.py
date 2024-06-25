#!/usr/bin/env python3
##### -*- coding: utf-8 -*-
"""
MLacc - Machine-Learning-based acceleration of spin-up

Copyright Laboratoire des Sciences du Climat et de l'Environnement (LSCE)
          Unite mixte CEA-CNRS-UVSQ

Code manager:
Daniel Goll, LSCE, dsgoll123@gmail.com

This software is developed by Yan Sun, Yilong Wang and Daniel Goll.......

This software is governed by the XXX license
XXXX <License content>
"""

from Tools import *
from DEF_Trunk import config

# added line
import numpy as np
import xarray
import subprocess

# print Python version
print(sys.version)

#
# Read configuration file
#

if len(sys.argv) < 2:
    dir_def = "DEF_Trunk/"
else:
    dir_def = sys.argv[1]

# Define task
itask = config.tasks

logfile = open(config.logfile, "w", buffering=1)
check.display("DEF directory: " + dir_def, logfile)
check.display("running task: %s" % str(itask), logfile)

# Define task
resultpath = config.results_dir
check.display("results are stored at: " + resultpath, logfile)

# Read list of variables
with open(dir_def + "varlist.json", "r") as f:
    varlist = json.loads(f.read())

# load stored results or start from scratch
if config.start_from_scratch:
    check.display("read from previous results...", logfile)
    packdata = xarray.load_dataset(resultpath + "packdata.nc")
else:
    check.display("MLacc start from scratch...", logfile)
    # initialize packaged data
    packdata = readvar(varlist, config, logfile)
    packdata.to_netcdf(resultpath + "packdata.nc")

# Define random seed
iseed = config.random_seed
random.seed(config.random_seed)

# added line, randomize with np also because Cluster_all
np.random.seed(iseed)

check.display("random seed = %i" % iseed, logfile)
# Define do leave-one-out crosee validation (loocv=1) or not (loocv=0)
loocv = config.leave_one_out_cv


if "1" in itask:
    #
    # test clustering
    dis_all = Cluster.Cluster_test(packdata, varlist, logfile)
    # added line
    np.random.seed(iseed)
    dis_all.dump(resultpath + "dist_all.npy")
    check.display(
        "test clustering done!\nResults have been stored as dist_all.npy", logfile
    )

    #
    # plot clustering results
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
        "test clustering results plotted!\nResults have been stored as dist_all.png",
        logfile,
    )
    # Run test of reproducibility for the task if yes
    if config.repro_test_task_1:
        subprocess.run(["python", "tests/task1_log.py"])
        check.display(
            "Task 1 reproducibility test results have been stored in tests_results.txt",
            logfile,
        )
    check.display("task 1: done", logfile)
if "2" in itask:
    #
    # clustering
    K = config.kmeans_clusters
    check.display("Kmean algorithm, K=%i" % K, logfile)
    IDx, IDloc, IDsel = Cluster.Cluster_all(packdata, varlist, K, logfile)
    np.savetxt(resultpath + "IDx.txt", IDx, fmt="%.2f")
    IDx.dump(resultpath + "IDx.npy")
    IDloc.dump(resultpath + "IDloc.npy")
    IDsel.dump(resultpath + "IDsel.npy")
    check.display("clustering done!\nResults have been stored as IDx.npy", logfile)

    #
    # plot clustering results
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
        "clustering results plotted!\nResults have been stored as ClustRes_PFT*.png",
        logfile,
    )
    # Run test of reproducibility for the task if yes
    if config.repro_test_task_2:
        subprocess.run(["python", "tests/task2_log.py"])
        check.display(
            "Task 2 reproducibility test results have been stored in tests_results.txt",
            logfile,
        )
    check.display("task 2: done", logfile)
if "3" in itask:
    #
    # build aligned forcing and aligned restart files
    check.check_file(resultpath + "IDx.npy", logfile)
    IDx = np.load(resultpath + "IDx.npy", allow_pickle=True)
    forcing.write(varlist, resultpath, IDx)
    # Run test of reproducibility for the task if yes
    if config.repro_test_task_3:
        subprocess.run(["python", "tests/task3_log.py"])
        subprocess.run(["python", "tests/task3_2_log.py"])
        check.display(
            "Task 3 reproducibility test results have been stored in tests_results.txt",
            logfile,
        )
    check.display("task 3: done", logfile)
if "4" in itask:
    # ML extrapolation

    var_pred_name1 = varlist["pred"]["allname"]
    var_pred_name2 = varlist["pred"]["allname_pft"]
    var_pred_name = var_pred_name1 + var_pred_name2
    # packdata.Nv_nopft = len(var_pred_name1)
    # packdata.Nv_total = len(var_pred_name)
    # packdata.var_pred_name = var_pred_name

    # Response variables
    Yvar = varlist["resp"]["variables"]
    responseY = Dataset(varlist["resp"]["sourcefile"], "r")

    check.check_file(resultpath + "IDx.npy", logfile)
    IDx = np.load(resultpath + "IDx.npy", allow_pickle=True)
    # generate PFT mask
    PFT_mask, PFT_mask_lai = genMask.PFT(
        packdata, varlist, varlist["PFTmask"]["pred_thres"]
    )

    # packdata.attrs['Nlat'] = np.trunc((90 - IDx[:, 0]) / packdata.lat_reso).astype(int)
    # packdata.attrs['Nlon'] = np.trunc((180 + IDx[:, 1]) / packdata.lon_reso).astype(int)
    packdata.attrs.update(
        Nv_nopft=len(var_pred_name1),
        Nv_total=len(var_pred_name),
        var_pred_name=var_pred_name,
        Nlat=np.trunc((90 - IDx[:, 0]) / packdata.lat_reso).astype(int),
        Nlon=np.trunc((180 + IDx[:, 1]) / packdata.lon_reso).astype(int),
    )
    labx = ["Y"] + var_pred_name + ["pft"]

    # copy the restart file to be modified
    targetfile = (
        varlist["resp"]["targetfile"]
        if "targetfile" in varlist["resp"]
        else varlist["resp"]["sourcefile"]
    )
    restfile = resultpath + targetfile.split("/")[-1]
    os.system("cp -f %s %s" % (targetfile, restfile))
    # add rights to manipulate file:
    os.chmod(restfile, 0o644)

    result = []
    for ipool in Yvar.keys():
        check.display("processing %s..." % ipool, logfile)
        res_df = ML.MLloop(
            packdata,
            ipool,
            logfile,
            varlist,
            labx,
            resultpath,
            loocv,
            restfile,
        )
        result.append(res_df)
    result_df = pd.concat(result, keys=Yvar.keys(), names=["comp"])
    result_df.to_csv(resultpath + "MLacc_results.csv")

    # we need to handle additional variables in the restart files but are not state variables of ORCHIDEE

    if "additional_vars" not in varlist["resp"]:
        # Handle the case where 'additional_vars' is not present
        print("We only modify true state variables of ORCHIDEE")
    else:
        additional_vars = varlist["resp"]["additional_vars"]

        for var in additional_vars:
            check.display("processing %s..." % var, logfile)
            restnc = Dataset(restfile, "a")
            # all variables derive from npp longterm prediction
            restvar = restnc["npp_longterm"]
            restvar1 = restnc[var]

            if var == "gpp_week" or var == "maxgppweek_lastyear" or var == "gpp_daily":
                tmpvar = restvar[:] * 2.0
            else:
                tmpvar = restvar[:]

            restvar1[:] = tmpvar
            restnc.close()
        # Run test of reproducibility for the task if yes
    if config.repro_test_task_4:
        subprocess.run(["python", "tests/task4_log.py"])
        subprocess.run(["python", "tests/task4_2_log.py"])
        check.display(
            "Task 4 reproducibility test results have been stored in tests_results.txt",
            logfile,
        )
    check.display("task 4: done", logfile)
if "5" in itask:
    Yvar = varlist["resp"]["variables"]
    for ipool in Yvar.keys():
        # if ipool!="litter":continue
        subpool_name = varlist["resp"]["pool_name_" + ipool]
        npfts = varlist["resp"]["npfts"]
        subLabel = varlist["resp"]["sub_item"]
        print(subLabel)
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
    check.display("task 5: done", logfile)
