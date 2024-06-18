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
    packdata, ivar, ipool, PFT_mask_lai, ipft, varname, ind, ii, labx, varlist, logfile
):
    check.display(
        "processing %s, variable %s, index %s (dim: %s)..."
        % (ipool, varname, ind, ii["dim_loop"]),
        logfile,
    )

    # extract data
    extr_var = extract_X.var(packdata, ipft)
    
    # extract PFT map
    pft_ny = extract_X.pft(packdata, PFT_mask_lai, ipft).reshape(len(packdata.Nlat), 1)

    # extract Y
    pool_arr = np.full(len(packdata.Nlat), np.nan)
    pool_map = np.squeeze(ivar)[
        tuple(i - 1 for i in ind)
    ]  # all indices start from 1, but python loop starts from 0
    pool_map[pool_map >= 1e18] = np.nan
    if "format" in varlist["resp"] and varlist["resp"]["format"] == "compressed":
        for cc in range(len(packdata.Nlat)):
            pool_arr[cc] = pool_map.flatten()[cc]
    else:
        for cc in range(len(packdata.Nlat)):
            pool_arr[cc] = pool_map[packdata.Nlat[cc], packdata.Nlon[cc]]

    extracted_Y = np.reshape(pool_arr, (len(packdata.Nlat), 1))
    extr_all = np.concatenate((extracted_Y, extr_var, pft_ny), axis=1)
    return DataFrame(extr_all, columns=labx)  # convert the array into dataframe


def combine_data(frames, keys):
    columns = frames[0].columns
    for frame in frames[1:]:
        if not columns.equals(frame.columns):
            raise ValueError("DataFrames have different columns")
    check_same = {}
    for col in columns:
        check_same[col] = all((frame[col] == frames[0][col]).dropna().all() for frame in frames)
    same_cols = [col for col, same in check_same.items() if same or col == 'pft']
    df = pd.concat([df.drop(columns=same_cols) for df in frames], keys=keys, axis=1)
    df.columns = [f"{c}_{k}" for k, c in df.columns]
    df = pd.concat([df, frames[0][same_cols]], axis=1)
    df = df.dropna().drop(columns=['pft'])  # delete pft=nan
    return df


def MLmap_multidim(
    packdata,
    combine_XY,
    ipool,
    PFT_mask,
    var_pred_name,
    varlist,
    labx,
    logfile,
    loocv,
    missVal,
):
    # need Yan Sun to modify it
    if "allname_type" in varlist["pred"].keys():
        col_type = labx.index(varlist["pred"]["allname_type"])
        type_val = varlist["pred"]["type_code"]
        combineXY = encode.en_code(combine_XY, col_type, type_val)
    else:
        col_type = "None"
        type_val = "None"
        combineXY = combine_XY
        
    Y = combineXY.filter(regex='^Y_')
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
    ) = train.training_BAT(X, Y, logfile, loocv)

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
            var_pred_name,
            model,
            col_type,
            type_val,
            var_pred_name,
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

    return MLeval.evaluation_map(Global_Predicted_Y_map, Y, PFT_mask)


def plot_eval_results(Global_Predicted_Y_map, ipool, pool_map, combineXY, predY_train, varname, ind, ii, ipft, PFT_mask, resultpath, logfile):
    
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
):
    var_pred_name1 = varlist["pred"]["allname"]
    var_pred_name2 = varlist["pred"]["allname_pft"]
    var_pred_name = var_pred_name1 + var_pred_name2

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

    comb_ds = defaultdict(list)

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
                        
                        comb_ds[ipool].append((
                            collect_data(
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
                                logfile,
                            ),
                            f"{varname}_{dim_ind[0]}_{dim_ind[1]}"
                        ))

                    # close&save netCDF file
                    restnc.close()

    results = []

    for ipool, vals in comb_ds.items():
        df = combine_data(*zip(*vals))
        df.to_csv(f"{resultpath}/{ipool}.csv")
    
        res = MLmap_multidim(
            packdata,
            df,
            ipool,
            PFT_mask,
            var_pred_name,
            varlist,
            labx,
            logfile,
            loocv,
            missVal,
        )
        results.append(res)
    
    return pd.concat(results, keys=comb_ds.keys(), names=["component"])
