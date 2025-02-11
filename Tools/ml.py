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


def mlmap_multidim(
    packdata,
    ivar,
    PFT_mask,
    PFT_mask_lai,
    ipool,
    ipft,
    varname,
    varlist,
    labx,
    ind,
    ii,
    leave_one_out_cv,
    smote_bat,
    restvar,
    missVal,
    alg,
    model_out_dir,
    seed,
):
    """
    Perform multi-dimensional machine learning mapping.

    This performs the following steps:
    * Extract data from the dataset.
    * Train a machine learning model.
    * Extrapolate the model to all global pixels.
    * Evaluate the model.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        ivar (numpy.ndarray): Array of response variable.
        PFT_mask (numpy.ndarray): Mask for Plant Functional Types.
        PFT_mask_lai (numpy.ndarray): Mask for Plant Functional Types based on LAI.
        ipool (str): Name of the current pool.
        ipft (int): Index of current Plant Functional Type.
        varname (str): Name of the current variable.
        varlist (dict): Dictionary of variable information.
        labx (list): List of column labels.
        ind (tuple): Index tuple for multi-dimensional variables.
        ii (dict): Dictionary containing dimension information.
        leave_one_out_cv (bool): Whether to use leave-one-out cross-validation.
        smote_bat (bool): Whether to use SMOTE balancing.
        restvar (numpy.ndarray): Restart variable.
        missVal (float): Missing value to use.
        alg (str): ML algorithm to use.
        model_out_dir (Path): Directory to save trained model output.
        seed (int): Random seed to ensure reproducibility.

    Returns:
        result (dict): Dictionary of evaluation results.
    """

    random.seed(seed)
    np.random.seed(seed)
    logfile = None
    check.display(
        "processing %s, variable %s, index %s (dim: %s)..."
        % (ipool, varname, ind, ii["dim_loop"]),
        logfile,
    )

    # 1. Extract data
    df_data, pool_map = extract_data(
        packdata, ivar, ipft, PFT_mask_lai, varlist, labx, ind
    )

    # 2. Train model
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

    # TODO: Need to check if the columns are the same in both files, need Yan Sun to modify it
    if "allname_type" in varlist["pred"].keys():
        col_type = labx.index(varlist["pred"]["allname_type"])
        type_val = varlist["pred"]["type_code"]
        combineXY = encode.encode(combine_XY, col_type, type_val)
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
    ) = train.training_bat(combineXY, logfile, leave_one_out_cv, smote_bat, seed, alg)

    # 3. Extrapolate
    Global_Predicted_Y_map = extrapolate_globally(
        model,
        predY_train,
        pool_map,
        packdata,
        ipft,
        PFT_mask,
        combine_XY,
        restvar,
        missVal,
        ind,
        col_type,
        type_val,
    )

    if "format" in varlist["resp"] and varlist["resp"]["format"] == "compressed":
        return None

    # 4. Evaluate
    if (PFT_mask[ipft - 1] > 0).any():
        return evaluate(
            ipool,
            ipft,
            varname,
            ind,
            ii,
            model,
            Global_Predicted_Y_map,
            pool_map,
            PFT_mask,
            varlist,
            model_out_dir,
        )
    else:
        check.display(
            "%s, variable %s, index %s (dim: %s) : NO DATA!"
            % (ipool, varname, ind, ii["dim_loop"]),
            logfile,
        )
        raise RuntimeError("No PFT mask found for PFT %i" % ipft)


def extract_data(packdata, ivar, ipft, PFT_mask_lai, varlist, labx, ind):
    """
    Extract data from the dataset.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        ivar (numpy.ndarray): Array of response variable.
        ipft (int): Index of current Plant Functional Type.
        PFT_mask_lai (numpy.ndarray): Mask for Plant Functional Types based on LAI.
        varlist (dict): Dictionary of variable information.
        labx (list): List of column labels.
        ind (tuple): Index tuple for multi-dimensional variables.

    Returns:
        DataFrame: DataFrame of extracted data.
        numpy.ma.core.MaskedArray: Map of target variables.
    """

    extr_var = extract_x.var(packdata, ipft)

    # Extract PFT map
    pft_ny = extract_x.pft(packdata, PFT_mask_lai, ipft)
    pft_ny = np.resize(pft_ny, (*extr_var.shape[:-1], 1))

    # Extract Y
    # All indices start from 1, but python loop starts from 0
    pool_map = np.squeeze(ivar)[tuple(i - 1 for i in ind)]
    pool_map[pool_map == 1e20] = np.nan
    # Y_map[ind[0]] = pool_map

    if "format" in varlist["resp"] and varlist["resp"]["format"] == "compressed":
        pool_arr = pool_map.flatten()
    else:
        pool_arr = pool_map[packdata.Nlat, packdata.Nlon]

    extracted_Y = np.resize(pool_arr, (*extr_var.shape[:-1], 1))

    extr_all = np.concatenate((extracted_Y, extr_var, pft_ny), axis=-1)
    extr_all = extr_all.reshape(-1, extr_all.shape[-1])

    return DataFrame(extr_all, columns=labx), pool_map


def extrapolate_globally(
    model,
    predY_train,
    pool_map,
    packdata,
    ipft,
    PFT_mask,
    combine_XY,
    restvar,
    missVal,
    ind,
    col_type,
    type_val,
):
    """
    Extrapolate predictions globally using the trained model.

    Args:
        model (sklearn.pipeline.Pipeline): Trained machine learning model.
        predY_train (numpy.ndarray): Predicted values from training set.
        pool_map (numpy.ma.core.MaskedArray): Map of target variables.
        packdata (xarray.Dataset): Dataset containing input variables.
        ipft (int): Index of current Plant Functional Type.
        PFT_mask (numpy.ndarray): Mask for Plant Functional Types.
        combine_XY (pandas.DataFrame): DataFrame of input variables.
        restvar (numpy.ndarray): Restart variable.
        missVal (float): Missing value to use.
        ind (tuple): Index tuple for multi-dimensional variables.
        col_type (str): Column name for encoding, or "None".
        type_val (int): Number of categories for encoding.

    Returns:
        Global_Predicted_Y_map: Predicted map of target variables.
    """

    if not model:
        # Only a single value
        predY = np.where(pool_map == pool_map, predY_train.iloc[0], np.nan)
        Global_Predicted_Y_map = predY
    else:
        Global_Predicted_Y_map, predY = mapglobe.extrp_global(
            packdata,
            ipft,
            PFT_mask,
            combine_XY.columns.drop("Y"),
            model,
            col_type,
            type_val,
        )
        # Modify restart file with extrapolated values
        pmask = np.nansum(PFT_mask, axis=0)
        pmask[np.isnan(pmask)] == 0
        # Set ocean pixel to missVal
        Pred_Y_out = np.where(pmask == 0, missVal, Global_Predicted_Y_map[:])
        # some pixel with nan remain, so set them zero
        Pred_Y_out = np.nan_to_num(Pred_Y_out)
        idx = (..., *[i - 1 for i in ind], slice(None), slice(None))
        restvar[idx] = Pred_Y_out
        # command = "restvar[...," + "%s," * len(ind) + ":,:]=Pred_Y_out[:]"
        # exec(command % tuple(ind - 1))
    return Global_Predicted_Y_map


def evaluate(
    ipool,
    ipft,
    varname,
    ind,
    ii,
    model,
    Global_Predicted_Y_map,
    pool_map,
    PFT_mask,
    varlist,
    model_out_dir,
):
    """
    Evaluate the machine learning model.

    Args:
        ipool (str): Name of the current pool.
        ipft (int): Index of current Plant Functional Type.
        varname (str): Name of the current variable.
        ind (tuple): Index tuple for multi-dimensional variables.
        ii (dict): Dictionary containing dimension information.
        model (sklearn.pipeline.Pipeline): Trained machine learning model.
        Global_Predicted_Y_map: Predicted map of target variables.
        pool_map: Map of target variables.
        PFT_mask: Mask for Plant Functional Types.
        varlist (dict): Dictionary of variable information.
        model_out_dir (Path): Directory to save trained model output.

    Returns:
        res (dict): Dictionary of evaluation results.

    """

    res = mleval.evaluation_map(Global_Predicted_Y_map, pool_map, ipft, PFT_mask)

    # In biomass the x-axis is the PFT, in the other pools it is the variable
    # varname looks something like "litter_01_ab" - this is unique
    # we need to assign a number to index based on (ipft) and (ind)
    # we then use this index as part of a multiindex ensuring uniqueness.
    if varname.startswith("biomass"):
        ipft = ind[0]
        index = int(re.search(r"\d+", varname)[0])
    elif (
        varname.startswith("carbon")
        or varname.startswith("nitrogen")
        or varname.startswith("phosphorus")
    ):
        ipft = int(re.search(r"\d+", varname)[0])
        # cnp = varname.split("_")[0]
        index = ind[0]
    elif varname.startswith("microbe") or varname.startswith("litter"):
        ipft = int(re.search(r"\d+", varname)[0])
        postfix = varname.split("_")[2]
        index = ind[0]
        j = ["ab", "be"].index(varname.split("_")[2])
        # ivar is numbered as so : (0 = ab, 1 = be, 2 = ab, 3 = be, 4 = ab, 5 = be)
        # this then matches up to the varlist
        # there should be a more elegant way to do this
        index = index * 2 + j - 1
    elif varname.startswith("npp") or varname.startswith("lai"):
        ipft = ind[0]
        index = None
    elif varname.startswith("lignin"):
        ipft = ind[0]
        index = ["ab", "be"].index(varname.split("_")[2])
        mat = ["struc", "wood"].index(varname.split("_")[1])
        index = 2 * index + mat  # [struc_ab, wood_ab, struc_be, wood_be]

    if type(model).__name__ == "Pipeline":
        alg = type(model.named_steps["estimator"]).__name__
    else:
        alg = type(model).__name__
    res["varname"] = varname
    res["ipft"] = ipft
    res["pft"] = f"PFT{ipft:02d}"
    res["ivar"] = index

    pools = ["som", "biomass", "litter", "microbe", "lignin"]

    if ipool in pools:
        # it would be good to eventually remove a dependency on the varlist
        # especially as is utilised for ordering purposes.
        res["var"] = varlist["resp"][f"pool_name_{ipool}"][index - 1]
    else:
        res["var"] = None
    res["dim"] = ii["dim_loop"][0]
    res["alg"] = alg

    if model_out_dir:
        model_out_dir = Path(model_out_dir)
        os.makedirs(model_out_dir, exist_ok=True)
        np.save(
            model_out_dir / f"{varname}_{index}_{ipft}.npy",
            dict(model=model, pred=Global_Predicted_Y_map),
            allow_pickle=True,
        )
    return res


def ml_loop(
    packdata,
    ipool,
    logfile,
    varlist,
    labx,
    config,
    restfile,
    alg,
    parallel,
    model_out_dir,
    seed,
):
    """
    Main loop for machine learning processing.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        ipool (str): Name of the current pool.
        logfile (file): File object for logging.
        varlist (dict): Dictionary of variable information.
        labx (list): List of column labels.
        config (module): module of config.
        restfile (str): Path to restart file.
        alg (str): ML algorithm to use.
        parallel (bool): Whether to run in parallel.
        save_model (bool): Option to save trained model output.
        seed (int): Random seed to ensure reproducibility.

    Returns:
        pandas.DataFrame: Results of machine learning evaluations.
    """
    responseY = Dataset(varlist["resp"]["sourcefile"], "r")
    PFT_mask, PFT_mask_lai = genmask.PFT(
        packdata, varlist, varlist["PFTmask"]["pred_thres"]
    )

    missVal = varlist["resp"]["missing_value"]

    inputs = []

    # Open restart file and select variable
    # - old comment suggested that memory was exceeded outside loop

    restnc = Dataset(restfile, "a")
    Yvar = varlist["resp"]["variables"][ipool]
    for ii in Yvar:
        for jj in ii["name_prefix"]:
            for kk in ii["loops"][ii["name_loop"]]:
                # Get response
                varname = jj + ("_%2.2i" % kk if kk else "") + ii["name_postfix"]
                if ii["name_loop"] == "pft":
                    ipft = kk
                ivar = responseY[varname]

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
                        inputs.append(
                            (
                                packdata,
                                ivar[:],
                                PFT_mask,
                                PFT_mask_lai,
                                ipool,
                                ipft,
                                varname,
                                varlist,
                                labx,
                                ind,
                                ii,
                                config.leave_one_out_cv,
                                config.smote_bat,
                                restvar[:],
                                missVal,
                                alg,
                                str(model_out_dir),
                                seed,
                            )
                        )
                    # Debugging
                    # if inputs:
                    #     break

                # Close netCDF file
    restnc.close()

    # Run the MLmap_multidim function in parallel or serial
    if parallel:
        import multiprocessing

        # multiprocessing.set_start_method("spawn", force=True)
        with ProcessPoolExecutor(max_workers=1) as executor:
            # Call the MLmap_multidim function with the arguments in inputs
            # Inputs is a list of tuples, each tuple is the arguments for the function
            # All inputs are collected in the result list
            print("Number of workers ", executor._max_workers)
            result = list(filter(None, executor.map(mlmap_multidim, *zip(*inputs))))
    else:
        # Serial processing
        result = []
        for input in inputs:
            if input:
                output = mlmap_multidim(*input)
                if output:  # Filter out None results
                    result.append(output)

    return pd.DataFrame(result).set_index(["ivar", "ipft"]).sort_index()
