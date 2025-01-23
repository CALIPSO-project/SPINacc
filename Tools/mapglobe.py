# =============================================================================================
# MLacc - Machine-Learning-based acceleration of spin-up
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


def extrp_global(packdata, ipft, PFTmask, XVarName, model, colum, Nm):
    """
    Extrapolate predictions globally using a trained model.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        ipft (int): index of PFT
        PFTmask (numpy.ndarray): Mask for Plant Functional Types.
        XVarName (list): List of input variable names.
        model: Trained machine learning model.
        colum (str): Column name for encoding, or "None".
        Nm (int): Number of categories for encoding.

    Returns:
        tuple:
            - Pred_Y_map (numpy.ndarray): Predicted map of target variables, masking nan pixels.
            - Pred_Y (numpy.ndarray): Predicted map of target variables, without masking.
    """
    if "year" in packdata.dims:
        packdata = packdata.mean("year", keep_attrs=True)
    # breakpoint()
    global_X_map = np.full((len(XVarName), packdata.nlat, packdata.nlon), np.nan)
    # PFTmask[np.isnan(PFTmask)]=0
    pmask = np.squeeze(PFTmask[ipft - 1][:])
    Pred_Y = np.full(PFTmask[0].shape, np.nan)
    # global metrics -> dataframe

    for ii in range(len(XVarName)):
        # if arr.ndim == 2:
        #     global_X_map[list(XVarName).index(arr.name)] = arr.values
        # else:
        #     global_X_map[list(XVarName).index(arr.name)] = arr[ipft - 1].values

        arr = packdata[XVarName[ii]]
        if arr.ndim == 2:
            global_X_map[ii] = arr.values
        else:
            global_X_map[ii] = arr[ipft - 1].values

        #    global_X_map=lc['global_X_map']
        das = global_X_map.transpose(1, 2, 0)
    for llat in range(packdata.nlat):
        Xllat = das[llat][:][:]
        # Xllat[np.isnan(Xllat)]=-9999
        Xtr = DataFrame(Xllat, columns=XVarName)
        ind = Xtr.index
        Xtr = Xtr.dropna()
        if len(Xtr) > 0:
            if colum != "None":
                Xtr_encode = encode.encode(Xtr, colum - 1, Nm)
            else:
                Xtr_encode = Xtr
            # Xtr.ix[:,colum]=(Xtr.ix[:,colum].astype(int)).astype(str)
            # Xtrr=pd.get_dummies(Xtr)
            Ym = DataFrame(model.predict(Xtr_encode))
            Ym.index = Xtr_encode.index
            Ymm = Ym.reindex(index=range(max(ind) + 1))
            Pred_Y[llat][:] = np.squeeze(Ymm)

    # laix=np.squeeze(packdata.LAI0[ipft-1][:][:])
    pmask[np.isnan(pmask)] = 0
    # pmask[laix<0.001]=0
    Pred_Y_map = Pred_Y * pmask
    # Pred_Y_map[land==1]=0

    # save map data
    # evaluation
    return Pred_Y_map, Pred_Y
