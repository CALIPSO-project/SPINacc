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


def MLmap_multidim(
    packdata,
    ivar,
    PFT_mask,
    PFT_mask_lai,
    ipool,
    ipft,
    logfile,
    varname,
    varlist,
    labx,
    ind,
    ii,
    resultpath,
    loocv,
    restvar,
    missVal,
    alg,
):
    """
    Perform multi-dimensional machine learning mapping.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        ivar (numpy.ndarray): Array of response variable.
        PFT_mask (numpy.ndarray): Mask for Plant Functional Types.
        PFT_mask_lai (numpy.ndarray): Mask for Plant Functional Types based on LAI.
        ipool (str): Name of the current pool.
        ipft (int): Index of current Plant Functional Type.
        logfile (file): File object for logging.
        varname (str): Name of the current variable.
        varlist (dict): Dictionary of variable information.
        labx (list): List of column labels.
        ind (tuple): Index tuple for multi-dimensional variables.
        ii (dict): Dictionary containing dimension information.
        resultpath (str): Path to store results.
        loocv (bool): Whether to perform leave-one-out cross-validation.
        restvar (numpy.ndarray): Restart variable.
        missVal (float): Missing value to use.
        alg (str): ML algorithm to use.

    Returns:
        tuple:
            - Global_Predicted_Y_map (numpy.ndarray): Globally predicted Y map.
            - model: Trained machine learning model.
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
    pool_map[pool_map >= 1e20] = np.nan
    # Y_map[ind[0]] = pool_map

    if "format" in varlist["resp"] and varlist["resp"]["format"] == "compressed":
        pool_arr = pool_map.flatten()
    else:
        pool_arr = pool_map[packdata.Nlat, packdata.Nlon]
    extracted_Y = np.resize(pool_arr, (*extr_var.shape[:-1], 1))

    extr_all = np.concatenate((extracted_Y, extr_var, pft_ny), axis=-1)
    extr_all = extr_all.reshape(-1, extr_all.shape[-1])

    df_data = DataFrame(extr_all, columns=labx)  # convert the array into dataframe
    combine_XY = df_data.dropna().drop(["pft"], axis=1)
    if len(combine_XY) == 0:
        check.display(
            "%s, variable %s, index %s (dim: %s) : NO DATA in training set!"
            % (ipool, varname, ind, ii["dim_loop"]),
            logfile,
        )
        if ind[-1] == ii["loops"][ii["dim_loop"][-1]][-1]:
            print(varname, ind)
        return None
    # need Yan Sun to modify it
    if "allname_type" in varlist["pred"].keys():
        col_type = labx.index(varlist["pred"]["allname_type"])
        type_val = varlist["pred"]["type_code"]
        combineXY = encode.en_code(combine_XY, col_type, type_val)
    else:
        col_type = "None"
        type_val = "None"
        combineXY = combine_XY
    # combine_XY=pd.get_dummies(combine_XY) # one-hot encoded
    (
        model,
        predY_train,
        loocv_R2,
        loocv_reMSE,
        loocv_slope,
        loocv_dNRMSE,
        loocv_sNRMSE,
        loocv_iNRMSE,
        loocv_f_SB,
        loocv_f_SDSD,
        loocv_f_LSC,
    ) = train.training_BAT(combineXY, logfile, loocv, alg)

    if not model:
        # only one value
        predY = np.where(pool_map == pool_map, predY_train.iloc[0], np.nan)
        Global_Predicted_Y_map = predY
    else:
        Global_Predicted_Y_map, predY = mapGlobe.extrp_global(
            packdata,
            ipft,
            PFT_mask,
            combine_XY.columns.drop("Y"),
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
        idx = (..., *[i - 1 for i in ind], slice(None), slice(None))
        restvar[idx] = Pred_Y_out
        # command = "restvar[...," + "%s," * len(ind) + ":,:]=Pred_Y_out[:]"
        # exec(command % tuple(ind - 1))

    if "format" in varlist["resp"] and varlist["resp"]["format"] == "compressed":
        return None

    if (PFT_mask[ipft - 1] > 0).any():
        res = MLeval.evaluation_map(Global_Predicted_Y_map, pool_map, ipft, PFT_mask)
        res["var"] = varname
        res["ind"] = ind[0]
        return res
        # check.display(
        #     "%s, variable %s, index %s (dim: %s) : R2=%.3f , RMSE=%.2f, slope=%.2f, reMSE=%.2f"
        #     % (ipool, varname, ind, ii["dim_loop"], R2, RMSE, slope, reMSE),
        #     logfile,
        # )
        # # save R2, RMSE, slope to txt files
        # # fx.write('%.2f' % R2+',')
        # # plot the results
        # fig = plt.figure(figsize=[12, 12])
        # # training dat
        # ax1 = plt.subplot(221)
        # ax1.scatter(combineXY.iloc[:, 0].values, predY_train)
        # # global dta
        # ax2 = plt.subplot(222)
        # #    predY=Global_Predicted_Y_map.flatten()
        # #    simuY=pool_map.flatten()
        # ax2.scatter(
        #     pool_map[PFT_mask[ipft - 1] > 0],
        #     Global_Predicted_Y_map[PFT_mask[ipft - 1] > 0],
        # )
        # xx = np.linspace(0, np.nanmax(pool_map), 10)
        # yy = np.linspace(0, np.nanmax(pool_map), 10)
        # ax2.text(
        #     0.1 * np.nanmax(pool_map),
        #     0.7 * np.nanmax(Global_Predicted_Y_map),
        #     "R2=%.2f" % R2,
        # )
        # ax2.text(
        #     0.1 * np.nanmax(pool_map),
        #     0.8 * np.nanmax(Global_Predicted_Y_map),
        #     "RMSE=%i" % RMSE,
        # )
        # ax2.plot(xx, yy, "k--")
        # ax2.set_xlabel("ORCHIDEE simulated")
        # ax2.set_ylabel("Machine-learning predicted")
        # ax3 = plt.subplot(223)
        # im = ax3.imshow(pool_map, vmin=0, vmax=0.8 * np.nanmax(pool_map))
        # ax3.set_title("ORCHIDEE simulated")
        # plt.colorbar(im, orientation="horizontal")
        # ax4 = plt.subplot(224)
        # im = ax4.imshow(Global_Predicted_Y_map, vmin=0, vmax=0.8 * np.nanmax(pool_map))
        # ax4.set_title("Machine-learning predicted")
        # plt.colorbar(im, orientation="horizontal")

        # fig.savefig(
        #     resultpath
        #     + "Eval_%s" % varname
        #     + "".join(
        #         ["_" + ii["dim_loop"][ll] + "%2.2i" % ind[ll] for ll in range(len(ind))]
        #         + [".png"]
        #     )
        # )
        # plt.close("all")
    else:
        check.display(
            "%s, variable %s, index %s (dim: %s) : NO DATA!"
            % (ipool, varname, ind, ii["dim_loop"]),
            logfile,
        )
    if ind[-1] == ii["loops"][ii["dim_loop"][-1]][-1]:
        raise Exception


def MLloop(
    packdata,
    ipool,
    logfile,
    varlist,
    labx,
    resultpath,
    loocv,
    restfile,
    alg,
):
    """
    Main loop for machine learning processing.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        ipool (str): Name of the current pool.
        logfile (file): File object for logging.
        varlist (dict): Dictionary of variable information.
        labx (list): List of column labels.
        resultpath (str): Path to store results.
        loocv (bool): Whether to perform leave-one-out cross-validation.
        restfile (str): Path to restart file.
        alg (str): ML algorithm to use.

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

    result = []

    Yvar = varlist["resp"]["variables"][ipool]
    for ii in Yvar:
        for jj in ii["name_prefix"]:
            for kk in ii["loops"][ii["name_loop"]]:
                varname = jj + ("_%2.2i" % kk if kk else "") + ii["name_postfix"]
                if ii["name_loop"] == "pft":
                    ipft = kk
                ivar = responseY[varname]

                # open restart file and select variable (memory is exceeded if open outside this loop)
                restnc = Dataset(restfile, "a")
                restvar = restnc[varname]

                if ii["dim_loop"] == ["null"] and ipft in ii["skip_loop"]["pft"]:
                    continue
                else:
                    index = itertools.product(
                        *[ii["loops"][ll] for ll in ii["dim_loop"]]
                    )
                    for ind in index:
                        dim_ind = tuple(zip(ii["dim_loop"], ind))
                        if "pft" in ii["dim_loop"]:
                            ipft = ind[ii["dim_loop"].index("pft")]
                        if ipft in ii["skip_loop"]["pft"]:
                            continue
                        result.append(
                            (
                                packdata,
                                ivar[:],
                                PFT_mask,
                                PFT_mask_lai,
                                ipool,
                                ipft,
                                None,  # logfile
                                varname,
                                varlist,
                                labx,
                                ind,
                                ii,
                                resultpath,
                                loocv,
                                restvar[:],
                                missVal,
                                alg,
                            )
                        )

                # close&save netCDF file
                restnc.close()

    # result = result[:3]
    with ThreadPoolExecutor() as pool:
        result = list(filter(None, pool.map(MLmap_multidim, *zip(*result))))

    return pd.DataFrame(result).set_index(["var", "ind"]).sort_index()
