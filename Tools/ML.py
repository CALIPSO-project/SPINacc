#
# Copyright Laboratoire des Sciences du Climat et de l'Environnement (LSCE)
#           Unite mixte CEA-CNRS-UVSQ
#
# Code manager:
# Daniel Goll, LSCE, <email>
#
# This software is developed by Yan Sun, Yilong Wang and Daniel Goll.......
#
# This software is governed by the XXX license
# XXXX <License content>
#
# =============================================================================================

from Tools import *
import itertools
from collections import defaultdict


def collect_data(
    packdata, ivar, ipool, PFT_mask_lai, ipft, varname, ind, ii, labx, varlist, Y_map, logfile
):
    """
    Collect and preprocess data for machine learning.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        ivar (numpy.ndarray): Array of response variable.
        ipool (str): Name of the current pool.
        PFT_mask_lai (numpy.ndarray): Mask for Plant Functional Types based on LAI.
        ipft (int): Index of current Plant Functional Type.
        varname (str): Name of the current variable.
        ind (tuple): Index tuple for multi-dimensional variables.
        ii (dict): Dictionary containing dimension information.
        labx (list): List of column labels.
        varlist (dict): Dictionary of variable information.
        Y_map (list): List to store Y maps.
        logfile (file): File object for logging.

    Returns:
        pandas.DataFrame: Collected and preprocessed data.
    """
    check.display(
        "processing %s, variable %s, index %s (dim: %s)..."
        % (ipool, varname, ind, ii["dim_loop"]),
        logfile,
    )

    # extract data
    extr_var = extract_X.var(packdata, ipft)

    # extract PFT map
    pft_ny = extract_X.pft(packdata, PFT_mask_lai, ipft)
    pft_ny = np.resize(pft_ny, (*extr_var.shape[:-1], 1))

    # extract Y
    pool_map = np.squeeze(ivar)[
        tuple(i - 1 for i in ind)
    ]  # all indices start from 1, but python loop starts from 0
    
    Y_map[ind[0]] = pool_map
    
    pool_map[pool_map >= 1e18] = np.nan
    if "format" in varlist["resp"] and varlist["resp"]["format"] == "compressed":
        pool_arr = pool_map.flatten()
    else:
        pool_arr = pool_map[packdata.Nlat, packdata.Nlon]
    extracted_Y = np.resize(pool_arr, (*extr_var.shape[:-1], 1))

    extr_all = np.concatenate((extracted_Y, extr_var, pft_ny), axis=-1)
    extr_all = extr_all.reshape(-1, extr_all.shape[-1])
    return DataFrame(extr_all, columns=labx)  # convert the array into dataframe


def combine_data(dics):
    """
    Combine data from multiple dictionaries into a single DataFrame.

    Args:
        dics (dict): Dictionary of data to combine.

    Returns:
        pandas.DataFrame: Combined data with NaN values and 'pft' column dropped.
    """
    df = pd.DataFrame(dics)
    df = df.dropna().drop(columns=['pft'])  # drop data according to the PFT mask
    return df


def MLmap_multidim(
    packdata,
    combine_XY,
    PFT_mask,
    varlist,
    labx,
    logfile,
    loocv,
    missVal,
    alg,
):
    """
    Perform multi-dimensional machine learning mapping.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        combine_XY (pandas.DataFrame): Combined X and Y data.
        PFT_mask (numpy.ndarray): Mask for Plant Functional Types.
        varlist (dict): Dictionary of variable information.
        labx (list): List of column labels.
        logfile (file): File object for logging.
        loocv (bool): Whether to perform leave-one-out cross-validation.
        missVal (float): Missing value to use.

    Returns:
        tuple: 
            - Global_Predicted_Y_map (numpy.ndarray): Globally predicted Y map.
            - model: Trained machine learning model.
    """
    # need Yan Sun to modify it
    if "allname_type" in varlist["pred"].keys():
        col_type = labx.index(varlist["pred"]["allname_type"])
        type_val = varlist["pred"]["type_code"]
        combineXY = encode.en_code(combine_XY, col_type, type_val)
    else:
        col_type = "None"
        type_val = "None"
        combineXY = combine_XY

    Y = combineXY.filter(regex="^Y_")
    X = combineXY.drop(columns=Y.columns)

    # combine_XY=pd.get_dummies(combine_XY) # one-hot encoded
    (
        model,
        predY_train,
        # loocv_R2,
        # loocv_reMSE,
        # loocv_slope,
        # loocv_dNRMSE,
        # loocv_sNRMSE,
        # loocv_iNRMSE,
        # loocv_f_SB,
        # loocv_f_SDSD,
        # loocv_f_LSC,
    ) = train.training_BAT(X, Y, logfile, loocv, alg)

    # pool_map = np.squeeze(ivar)[
    #     tuple(i - 1 for i in ind)
    # ]  # all indices start from 1, but python loop starts from 0
    # pool_map[pool_map >= 1e15] = np.nan

    if not model:
        # only one value
        # predY = np.where(pool_map == pool_map, predY_train[0], np.nan)
        Global_Predicted_Y_map = predY_train
    else:
        Global_Predicted_Y_map, predY = mapGlobe.extrp_global(
            packdata,
            PFT_mask,
            X.columns,
            model,
            col_type,
            type_val,
        )
        # write to restart file
        pmask = np.nansum(PFT_mask, axis=0)
        pmask[np.isnan(pmask)] == 0
        # set ocean pixel to missVal
        Pred_Y_out = np.where(pmask == 0, missVal, Global_Predicted_Y_map[:])
        # some pixel with nan remain, so set them zero
        Pred_Y_out = np.nan_to_num(Pred_Y_out)
        # idx = (..., *[i - 1 for i in ind], slice(None), slice(None))
        # restvar[idx] = Pred_Y_out
        # command = "restvar[...," + "%s," * len(ind) + ":,:]=Pred_Y_out[:]"
        # exec(command % tuple(ind - 1))

    if "format" in varlist["resp"] and varlist["resp"]["format"] == "compressed":
        return None

    return Global_Predicted_Y_map, model


def plot_eval_results(
    Global_Predicted_Y_map,
    ipool,
    pool_map,
    combineXY,
    predY_train,
    varname,
    ind,
    ii,
    ipft,
    PFT_mask,
    resultpath,
    logfile,
):
    # evaluation
    R2, RMSE, slope, reMSE, dNRMSE, sNRMSE, iNRMSE, f_SB, f_SDSD, f_LSC = (
        MLeval.evaluation_map(Global_Predicted_Y_map, pool_map, ipft, PFT_mask)
    )
    check.display(
        "%s, variable %s, index %s (dim: %s) : R2=%.3f , RMSE=%.2f, slope=%.2f, reMSE=%.2f"
        % (ipool, varname, ind, ii["dim_loop"], R2, RMSE, slope, reMSE),
        logfile,
    )
    # save R2, RMSE, slope to txt files
    # fx.write('%.2f' % R2+',')
    # plot the results
    fig = plt.figure(figsize=[12, 12])
    # training dat
    ax1 = plt.subplot(221)
    ax1.scatter(combineXY.iloc[:, 0].values, predY_train)
    # global dta
    ax2 = plt.subplot(222)
    #    predY=Global_Predicted_Y_map.flatten()
    #    simuY=pool_map.flatten()
    ax2.scatter(
        pool_map[PFT_mask[ipft - 1] > 0],
        Global_Predicted_Y_map[PFT_mask[ipft - 1] > 0],
    )
    xx = np.linspace(0, np.nanmax(pool_map), 10)
    yy = np.linspace(0, np.nanmax(pool_map), 10)
    ax2.text(
        0.1 * np.nanmax(pool_map),
        0.7 * np.nanmax(Global_Predicted_Y_map),
        "R2=%.2f" % R2,
    )
    ax2.text(
        0.1 * np.nanmax(pool_map),
        0.8 * np.nanmax(Global_Predicted_Y_map),
        "RMSE=%i" % RMSE,
    )
    ax2.plot(xx, yy, "k--")
    ax2.set_xlabel("ORCHIDEE simulated")
    ax2.set_ylabel("Machine-learning predicted")
    ax3 = plt.subplot(223)
    im = ax3.imshow(pool_map, vmin=0, vmax=0.8 * np.nanmax(pool_map))
    ax3.set_title("ORCHIDEE simulated")
    plt.colorbar(im, orientation="horizontal")
    ax4 = plt.subplot(224)
    im = ax4.imshow(Global_Predicted_Y_map, vmin=0, vmax=0.8 * np.nanmax(pool_map))
    ax4.set_title("Machine-learning predicted")
    plt.colorbar(im, orientation="horizontal")

    fig.savefig(
        resultpath
        + "Eval_%s" % varname
        + "".join(
            ["_" + ii["dim_loop"][ll] + "%2.2i" % ind[ll] for ll in range(len(ind))]
            + [".png"]
        )
    )
    plt.close("all")


##@param[in]   packdata               packaged data
##@param[in]   logfile                logfile
def MLloop(
    packdata,
    logfile,
    varlist,
    labx,
    resultpath,
    loocv,
    restfile,
    alg
):
    """
    Main loop for machine learning processing.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        logfile (file): File object for logging.
        varlist (dict): Dictionary of variable information.
        labx (list): List of column labels.
        resultpath (str): Path to store results.
        loocv (bool): Whether to perform leave-one-out cross-validation.
        restfile (str): Path to restart file.

    Returns:
        pandas.DataFrame: Results of machine learning evaluations.
    """
    responseY = Dataset(varlist["resp"]["sourcefile"], "r")
    PFT_mask, PFT_mask_lai = genMask.PFT(
        packdata, varlist, varlist["PFTmask"]["pred_thres"]
    )

    # We can't do it here: we would overwrite the file repeatidly loosing the information we wrote before
    # it's done now in main.py
    # Copy restart file template (might have to be changed)
    #  restfile=resultpath+varlist['resp']['sourcefile'].split('/')[-1]
    #  os.system('cp -f %s %s'%(varlist['resp']['sourcefile'],restfile))
    missVal = varlist["resp"]["missing_value"]
    Yvar = varlist["resp"]["variables"]

    comb_ds = defaultdict(dict)
    Y_maps = defaultdict(dict)
    # dup_cols = [k for k, v in packdata.items() if "veget" not in v.dims] + ["pft"]

    for ipool, iis in Yvar.items():
        check.display("processing %s..." % ipool, logfile)

        for ii in iis:
            for jj in ii["name_prefix"]:
                for kk in ii["loops"][ii["name_loop"]]:
                    varname = jj + ("_%2.2i" % kk if kk else "") + ii["name_postfix"]
                    if ii["name_loop"] == "pft":
                        ipft = kk
                    ivar = responseY[varname]
                    
                    # open restart file and select variable (memory is exceeded if open outside this loop)
                    restnc = Dataset(restfile, "a")
                    # restvar = restnc[varname]

                    index = itertools.product(
                        *[ii["loops"][ll] for ll in ii["dim_loop"]]
                    )
                    for ind in index:
                        if "pft" in ii["dim_loop"]:
                            ipft = ind[ii["dim_loop"].index("pft")]
                        if ipft in ii["skip_loop"]["pft"]:
                            continue

                        dim_ind, = zip(ii["dim_loop"], ind)
                        
                        df = collect_data(
                            packdata,
                            ivar,
                            ipool,
                            PFT_mask_lai,
                            ipft,
                            varname,
                            ind,
                            ii,
                            labx,
                            varlist,
                            Y_maps[ipool, varname],
                            logfile,
                        )
                        
                        for k, s in df.items():
                            dic = comb_ds[ipool, varname]
                            if k == "Y":
                                k = f"Y_{dim_ind[0]}_{dim_ind[1]}"
                            elif k == "pft":
                                if k in dic:
                                    s = dic[k].combine_first(s)
                            elif "veget" in packdata[k].dims:
                                k = f"{k}_{varname}"
                            dic[k] = s

                    # close&save netCDF file
                    restnc.close()
                    
    results = dict()

    for (ipool, var), dics in comb_ds.items():
        df = combine_data(dics)
        df.to_csv(f"{resultpath}/{ipool}_{var}.csv", index=False)

        Y = df.filter(regex="^Y_")
        X = df.drop(columns=Y.columns)
        
        for label in Y.columns:
            pred_Y_map, model = MLmap_multidim(
                packdata,
                df[[*X.columns, label]],
                PFT_mask,
                varlist,
                labx,
                logfile,
                loocv,
                missVal,
                alg
            )
            ind = int(label.split("_")[-1])
            Y_map = Y_maps[ipool, var][ind]
            res = MLeval.evaluation_map(pred_Y_map, Y_map)
            results[ipool, var, ind] = res
        
    return pd.DataFrame(results).T
