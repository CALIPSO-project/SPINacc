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


def detect_restart_type(restfile):
    """
    Detect whether a restart file is structured (lat/lon grid) or unstructured (cell-based).

    Args:
        restfile (str): Path to the restart file.

    Returns:
        str: "unstructured" if the file has a 'cell' dimension, otherwise "structured".
    """
    with Dataset(restfile, "r") as nc:
        if "cell" in nc.dimensions:
            return "unstructured"
    return "structured"


def get_unstructured_cell_indices(restfile, packdata):
    """
    Get global lat/lon grid indices for each cell in an unstructured restart file.

    The lat/lon coordinates of each cell are read from the file and mapped to
    integer row/column indices in packdata's global regular grid.

    Args:
        restfile (str): Path to unstructured restart file.
        packdata (xarray.Dataset): Dataset with lat_reso and lon_reso attributes.

    Returns:
        tuple: (Nlat, Nlon) integer arrays of global grid indices for each cell.
    """
    with Dataset(restfile, "r") as nc:
        lat_var = next(
            (v for v in ["nav_lat", "lat", "latitude"] if v in nc.variables), None
        )
        lon_var = next(
            (v for v in ["nav_lon", "lon", "longitude"] if v in nc.variables), None
        )
        if lat_var is None or lon_var is None:
            raise ValueError(
                f"Cannot find lat/lon variables in unstructured restart file: {restfile}"
            )
        cell_lats = np.asarray(nc.variables[lat_var][:])
        cell_lons = np.asarray(nc.variables[lon_var][:])
    Nlat = np.trunc((90 - cell_lats) / packdata.lat_reso).astype(int)
    Nlon = np.trunc((180 + cell_lons) / packdata.lon_reso).astype(int)
    return Nlat, Nlon


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
    rest_Nlat=None,
    rest_Nlon=None,
):
    """
    Perform multi-dimensional machine learning mapping.

    This performs the following steps:
    * Extract data from the dataset.
    * Train a machine learning model.
    * Extrapolate the model to all global pixels.
    * Write the extrapolated values to the restart file.
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
        rest_Nlat (numpy.ndarray or None): Global lat indices for unstructured restart cells.
        rest_Nlon (numpy.ndarray or None): Global lon indices for unstructured restart cells.

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
    Global_Predicted_Y_map, Pred_Y_out, idx = extrapolate_globally(
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
        rest_Nlat,
        rest_Nlon,
    )

    if "format" in varlist["resp"] and varlist["resp"]["format"] == "compressed":
        return None

    # For unstructured format: predictions are written back but global evaluation is skipped
    # because pool_map contains only the training cells, not the full global grid.
    if "format" in varlist["resp"] and varlist["resp"]["format"] == "unstructured":
        return (None, Pred_Y_out, idx, varname)

    # 4. Evaluate
    if (PFT_mask[ipft - 1] > 0).any():
        return (
            evaluate(
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
            ),
            Pred_Y_out,
            idx,
            varname,
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

    if "format" in varlist["resp"] and varlist["resp"]["format"] in ("compressed", "unstructured"):
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
    rest_Nlat=None,
    rest_Nlon=None,
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
        rest_Nlat (numpy.ndarray or None): Global lat indices for unstructured restart cells.
            If provided (unstructured restart), predictions are extracted at these positions.
        rest_Nlon (numpy.ndarray or None): Global lon indices for unstructured restart cells.

    Returns:
        Global_Predicted_Y_map: Predicted map of target variables (always global nlat x nlon).
        Pred_Y_out: Values to write back to the restart file.
        idx: Index tuple for writing into the restart variable.
    """
    Pred_Y_out = None
    idx = None
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
        if rest_Nlat is not None:
            # Unstructured restart: extract predictions at each cell's global position.
            # The global extrapolation (nlat x nlon) is indexed by the cell coordinates.
            Pred_Y_out = np.nan_to_num(Global_Predicted_Y_map[rest_Nlat, rest_Nlon])
            idx = (..., *[i - 1 for i in ind], slice(None))
        else:
            # Structured restart: write the full global lat/lon prediction map.
            # Modify restart file with extrapolated values
            pmask = np.nansum(PFT_mask, axis=0)
            pmask[np.isnan(pmask)] == 0
            # Set ocean pixel to missVal
            Pred_Y_out = np.where(pmask == 0, missVal, Global_Predicted_Y_map[:])
            # some pixel with nan remain, so set them zero
            Pred_Y_out = np.nan_to_num(Pred_Y_out)
            idx = (..., *[i - 1 for i in ind], slice(None), slice(None))
    return Global_Predicted_Y_map, Pred_Y_out, idx


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

    # Detect restart file type once and get cell indices for unstructured files.
    # For unstructured restart files (cell dimension), rest_Nlat/rest_Nlon map each
    # cell to its position in the global regular grid used for feature extrapolation.
    # For structured restart files (lat/lon grid), these are None and the full global
    # prediction map is written back directly.
    rest_type = detect_restart_type(restfile)
    rest_Nlat, rest_Nlon = None, None
    if rest_type == "unstructured":
        rest_Nlat, rest_Nlon = get_unstructured_cell_indices(restfile, packdata)

    # Open restart file and select variable
    # - old comment suggested that memory was exceeded outside loop

    restnc = Dataset(restfile, "a")
    result = []

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
                        # inputs.append(
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
                                rest_Nlat,
                                rest_Nlon,
                            )
                        )

    # Close files
    responseY.close()
    restnc.close()

    # # Run the MLmap_multidim function in parallel or serial
    if parallel:
        with ProcessPoolExecutor() as executor:
            # Call the MLmap_multidim function with the arguments in inputs
            # Inputs is a list of tuples, each tuple is the arguments for the function
            # All inputs are collected in the result list
            print("Number of workers ", executor._max_workers)
            result, Pred_Y_out, idx, varname = zip(
                *filter(None, executor.map(mlmap_multidim, *zip(*inputs)))
            )
            rest_state = list(zip(varname, idx, Pred_Y_out))
            all_result = [r for r in result if r is not None]

    else:
        # Serial processing
        all_result = []
        rest_state = []
        for input in inputs:
            if input:
                output = mlmap_multidim(*input)
                if output is None:
                    continue
                result, Pred_Y_out, idx, varname = output
                if result is not None:
                    all_result.append(result)
                rest_state.append((varname, idx, Pred_Y_out))

    # Modify restart file

    restnc = Dataset(restfile, "a")
    if rest_state:
        for varname, idx, Pred_Y_out in rest_state:
            if Pred_Y_out is not None:
                restnc[varname][idx] = Pred_Y_out
    restnc.close()

    return pd.DataFrame(all_result).set_index(["ivar", "ipft"]).sort_index()
