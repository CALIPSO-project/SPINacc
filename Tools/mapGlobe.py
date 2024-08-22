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


def extrp_global(packdata, PFTmask, XVarName, model, colum, Nm):
    """
    Extrapolate predictions globally using a trained model.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
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
    global_X_map = []
    # PFTmask[np.isnan(PFTmask)]=0
    # global metrics -> dataframe
    try:
        packdata = packdata.mean(dim=('year', 'month'))
    except ValueError:
        pass
    for varname in XVarName:
        if varname in packdata.data_vars:
            x = packdata[varname].values
        else:
            varname, ipft = re.match(r"(\w+)?_\w+?_(\d+)", varname).groups()
            ipft = int(ipft) - 1
            x = packdata[varname].sel(veget=ipft).values
        global_X_map.append(x)
    das = np.stack(global_X_map, axis=2)
    Xtr = DataFrame(das.reshape(-1, das.shape[-1]), columns=XVarName)
    if colum != "None":
        Xtr_encode = encode.en_code(Xtr, colum - 1, Nm)
    else:
        Xtr_encode = Xtr
    Xtr_encode = Xtr_encode.dropna()
    Ym = DataFrame(model.predict(Xtr_encode), index=Xtr_encode.index)
    Ymm = Ym.reindex(index=Xtr.index)
    Pred_Y = Ymm.values.reshape(*das.shape[:-1], -1).transpose(2, 0, 1)
    
    # laix=np.squeeze(packdata.LAI0[ipft-1][:][:])
    # pmask = np.squeeze(PFTmask[ipft - 1][:])
    # pmask[np.isnan(pmask)] = 0
    pmask = 1
    # pmask[laix<0.001]=0
    Pred_Y_map = Pred_Y * pmask
    # Pred_Y_map[land==1]=0

    # save map data
    # evaluation
    return Pred_Y_map, Pred_Y
