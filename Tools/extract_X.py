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


##@param[in]   packdata               packaged data
##@param[in]   var_ind                index of variable
##@param[in]   VarName                list of variables' names
##@retval      VarN                   variable values of selected pixels
def extract_X(packdata, var_ind, VarName):
    Nlat = packdata.Nlat
    Nlon = packdata.Nlon
    varN = np.full(len(Nlat), np.nan)
    var_data = packdata[VarName[var_ind]]
    for cc in range(0, len(Nlat)):
        varN[cc] = var_data[Nlat[cc], Nlon[cc]]
    return varN


##@param[in]   packdata               packaged data
##@param[in]   var_ind                index of variable
##@param[in]   VarName                list of variables' names
##@param[in]   px                     index of PFT
##@retval      VarN                   variable values of selected pixels
def extract_XN(packdata, var_ind, VarName, px):
    Nlat = packdata.Nlat
    Nlon = packdata.Nlon
    varN = np.full(len(Nlat), np.nan)
    var_data = packdata[VarName[var_ind]]
    var_pft_map = np.squeeze(var_data[px - 1][:][:])
    for cc in range(0, len(Nlat)):
        varN[cc] = var_pft_map[Nlat[cc], Nlon[cc]]
    return varN


##@param[in]   packdata               packaged data
##@param[in]   PFT_mask               PFT mask
##@param[in]   px                     index of PFT
##@retval      VarN                   variable values of selected pixels
def pft(packdata, PFT_mask, px):
    Nlat = packdata.Nlat
    Nlon = packdata.Nlon
    varN = np.full(len(Nlat), np.nan)
    for cc in range(0, len(Nlat)):
        varN[cc] = PFT_mask[px - 1, Nlat[cc], Nlon[cc]]
    return varN


##@param[in]   packdata               packaged data
##@param[in]   ipft                   ith pft
##@retval      extr_var               extracked data
def var(packdata, ipft):
    extr_var = np.zeros(shape=(len(packdata.Nlat), 0))
    for indx in range(packdata.Nv_total):
        if indx < packdata.Nv_nopft:
            extracted_var = np.reshape(
                extract_X(packdata, indx, packdata.var_pred_name),
                (len(packdata.Nlat), 1),
            )
            extr_var = np.concatenate((extr_var, extracted_var), axis=1)
        else:
            extracted_var = np.reshape(
                extract_XN(packdata, indx, packdata.var_pred_name, ipft),
                (len(packdata.Nlat), 1),
            )
            extr_var = np.concatenate((extr_var, extracted_var), axis=1)
    return extr_var
