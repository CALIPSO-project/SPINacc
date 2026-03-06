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

def remap_unstructured_to_structured(source_var, packdata, missVal=np.nan):
    """
    Remap pseudo-unstructured variable to structured grid,
    handling extra dimensions and cell dimension not at the last axis.
    """

    source_var = np.asarray(source_var)
    n_cells = packdata.Nlat.size
    print(n_cells)

    # Find axis corresponding to the cell dimension
    # Prefer the last dimension matching n_cells, as cell dimensions typically appear last.
    matching_axes = [ax for ax, size in enumerate(source_var.shape) if size == n_cells]
    if not matching_axes:
        raise RuntimeError(
            f"Cannot find dimension matching n_cells={n_cells} in source_var.shape={source_var.shape}"
        )
    cell_axis = matching_axes[-1]

    # Move the cell axis to the last position
    if cell_axis != len(source_var.shape) - 1:
        source_var = np.moveaxis(source_var, cell_axis, -1)

    # Now the last dimension is the spatial cell dimension
    structured_shape = source_var.shape[:-1] + (packdata.nlat, packdata.nlon)
    structured = np.full(structured_shape, missVal, dtype=source_var.dtype)

    # Flatten spatial indices
    flat_idx = packdata.Nlat * packdata.nlon + packdata.Nlon
    structured_flat = structured.reshape(*source_var.shape[:-1], -1)

    # Assign cell values to global structured grid
    structured_flat[..., flat_idx] = source_var[..., packdata.cell_idx]

    # Reshape back to final structured shape
    structured = structured_flat.reshape(structured_shape)
    return structured


def detect_grid_type(ncfile):
    """
    Detect whether a netCDF file is on a structured (lat/lon grid) or unstructured
    (cell-based) grid.

    A file is considered unstructured if it has:
    - a 'cell' dimension (true unstructured), OR
    - a 'y' dimension with multiple cells and an 'x' dimension of size 1
      (pseudo-unstructured, as produced by Step 3).

    Args:
        ncfile (str): Path to the netCDF file.

    Returns:
        str: "unstructured" if the file is in unstructured or pseudo-unstructured
             format, otherwise "structured".
    """
    with Dataset(ncfile, "r") as nc:
        if "cell" in nc.dimensions:
            return "unstructured"
        # Pseudo-unstructured: y dimension with multiple cells, x dimension of size 1
        if (
            "y" in nc.dimensions
            and "x" in nc.dimensions
            and len(nc.dimensions["x"]) == 1
            and len(nc.dimensions["y"]) > 1
        ):
            return "unstructured"
    return "structured"


def _build_cell_idx_map(sourcefile, packdata):
    """
    Build a mapping from training-pixel global-grid indices (Nlat, Nlon) to the
    cell position in an unstructured source file.

    The unstructured source file stores only a subset of pixels (the ones selected
    during clustering).  This function reads the lat/lon coordinates stored in that
    file, converts them to global-grid indices using the same formula used in
    main.py for packdata.Nlat/Nlon, and returns an index array so that

        pool_map.ravel()[cell_idx[j]]

    gives the value that belongs to training pixel j (identified by
    packdata.Nlat[j] / packdata.Nlon[j]).

    Args:
        sourcefile (str): Path to the unstructured source file.
        packdata (xarray.Dataset): Dataset with attrs Nlat, Nlon, lat_reso, lon_reso.

    Returns:
        numpy.ndarray: Integer array of shape (len(packdata.Nlat),) with the cell
            index in the source file for each training pixel.

    Raises:
        RuntimeError: If lat/lon coordinate variables cannot be found in the source
            file, or if a training pixel is absent from the source file.
    """
    lat_names = ["nav_lat", "lat", "latitude"]
    lon_names = ["nav_lon", "lon", "longitude"]

    with Dataset(sourcefile, "r") as nc:
        cell_lats = None
        cell_lons = None
        for name in lat_names:
            if name in nc.variables:
                cell_lats = np.squeeze(nc.variables[name][:])
                break
        for name in lon_names:
            if name in nc.variables:
                cell_lons = np.squeeze(nc.variables[name][:])
                break

    if cell_lats is None or cell_lons is None:
        raise RuntimeError(
            "Could not find lat/lon coordinate variables in unstructured source file"
        )

    # Convert cell lat/lon to global grid indices (same formula as in main.py)
    cell_ilats = np.trunc((90 - cell_lats) / packdata.lat_reso).astype(int)
    cell_ilons = np.trunc((180 + cell_lons) / packdata.lon_reso).astype(int)

    # Build reverse mapping: (ilat, ilon) -> cell index in the source file
    cell_map = {
        (int(ilat), int(ilon)): i
        for i, (ilat, ilon) in enumerate(
            zip(cell_ilats.ravel(), cell_ilons.ravel())
        )
    }

    # For each training pixel, look up its cell index in the source file
    try:
        cell_idx = np.array(
            [
                cell_map[(int(nlat), int(nlon))]
                for nlat, nlon in zip(packdata.Nlat, packdata.Nlon)
            ]
        )
    except KeyError as e:
        raise RuntimeError(
            f"Training pixel with global-grid index {e} was not found in the "
            "unstructured source file. The source file may not contain all "
            "training pixels."
        ) from e

    return cell_idx


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
        missVal,
        ind,
        col_type,
        type_val,
    )

    if "format" in varlist["resp"] and varlist["resp"]["format"] == "compressed":
        return None

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

    resp_format = varlist["resp"].get("format", "regular")
    if resp_format == "compressed":
        pool_arr = pool_map.flatten()
    elif resp_format == "unstructured":
    #    # pool_map is a flat 1-D array (n_cells,) from the unstructured source file.
    #    # packdata.cell_idx maps each training pixel to its position in that array.
    #    pool_arr = pool_map.ravel()[packdata.cell_idx]
    #else:
    #    # "regular" or any other structured format
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
        missVal (float): Missing value to use.
        ind (tuple): Index tuple for multi-dimensional variables.
        col_type (str): Column name for encoding, or "None".
        type_val (int): Number of categories for encoding.

    Returns:
        Global_Predicted_Y_map: Predicted map of target variables.
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
        # Modify restart file with extrapolated values
        pmask = np.nansum(PFT_mask, axis=0)
        pmask[np.isnan(pmask)] = 0
        # Set ocean pixel to missVal
        Pred_Y_out = np.where(pmask == 0, missVal, Global_Predicted_Y_map[:])
        # some pixel with nan remain, so set them zero
        Pred_Y_out = np.nan_to_num(Pred_Y_out)
        idx = (..., *[i - 1 for i in ind], slice(None), slice(None))
        # breakpoint()
        # restvar[idx] = Pred_Y_out
        # command = "restvar[...," + "%s," * len(ind) + ":,:]=Pred_Y_out[:]"
        # exec(command % tuple(ind - 1))
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
    sourcefile,
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
        restfile (str): Path to restart file ( global structured grid )
        sourcefile (str): Path to restart file with training data ( grid type can vary )
        alg (str): ML algorithm to use.
        parallel (bool): Whether to run in parallel.
        model_out_dir (Path): Directory in which to save trained model output.
        seed (int): Random seed to ensure reproducibility.

    Returns:
        pandas.DataFrame: Results of machine learning evaluations.
    """

    # check for grid type in source file
    rest_type = detect_grid_type(sourcefile)
    # Map the varlist format name to the expected detected type.
    # "regular" and "compressed" both correspond to a structured global grid;
    # "unstructured" corresponds to either a true unstructured grid (cell dimension)
    # or a pseudo-unstructured grid (y × x=1 layout produced by Step 3).
    resp_format = varlist["resp"].get("format", "regular")
    expected_type = "unstructured" if resp_format == "unstructured" else "structured"
    if rest_type != expected_type:
        raise RuntimeError(
            f"source file does not correspond to expected grid format, but is {rest_type}"
        )

    # For unstructured source files, build a mapping from the training-pixel
    # global-grid indices (Nlat, Nlon) to the cell position in the source file,
    # and store it as a packdata attribute so that extract_data can use it.
    if rest_type == "unstructured":
        cell_idx = _build_cell_idx_map(sourcefile, packdata)
        packdata = packdata.assign_attrs(cell_idx=cell_idx)

    responseY = Dataset(sourcefile, "r")
    # print(responseY)

    PFT_mask, PFT_mask_lai = genmask.PFT(
        packdata, varlist, varlist["PFTmask"]["pred_thres"]
    )

    missVal = varlist["resp"]["missing_value"]

    inputs = []

    # Open restart file with the training data and select variables
    # - old comment suggested that memory was exceeded outside loop

    result = []

    Yvar = varlist["resp"]["variables"][ipool]
    for ii in Yvar:
        for jj in ii["name_prefix"]:
            for kk in ii["loops"][ii["name_loop"]]:
                # Get response
                varname = jj + ("_%2.2i" % kk if kk else "") + ii["name_postfix"]
                if ii["name_loop"] == "pft":
                    ipft = kk
                #ivar = responseY[varname]
                #ivar = responseY[varname][:].squeeze()

                if rest_type == "unstructured":
                    ivar = remap_unstructured_to_structured(responseY[varname][:], packdata, missVal)
                else:
                    ivar = responseY[varname]

                print(varname, np.shape(ivar))
                
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
                                missVal,
                                alg,
                                str(model_out_dir),
                                seed,
                            )
                        )

    # Close files
    responseY.close()

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
            all_result = list(result)

    else:
        # Serial processing
        all_result = []
        rest_state = []
        for input in inputs:
            if input:
                result, Pred_Y_out, idx, varname = mlmap_multidim(*input)
                if result:  # Filter out None results
                    all_result.append(result)
                    rest_state.append((varname, idx, Pred_Y_out))

    # Modify restart file ( global structured grid )

    # check for grid type in source file
    rest_type = detect_grid_type(restfile)
    if rest_type != "structured":
        raise RuntimeError(
            f"target file does not correspond to expected grid format, but is {rest_type}"
        )
    # print(rest_type)

    restnc = Dataset(restfile, "a")
    if rest_state:
        for varname, idx, Pred_Y_out in rest_state:
            if Pred_Y_out is not None:
                restnc[varname][idx] = Pred_Y_out
    restnc.close()

    return pd.DataFrame(all_result).set_index(["ivar", "ipft"]).sort_index()
