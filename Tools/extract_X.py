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
##@param[in]   auxil                  auxiliary data
##@param[in]   var_ind                index of variable
##@param[in]   VarName                list of variables' names
##@retval      VarN                   variable values of selected pixels
def extract_X(packdata, auxil, var_ind, VarName):
    Nlat = auxil.Nlat
    Nlon = auxil.Nlon
    varN = np.full(len(Nlat), np.nan)
    var_data = packdata[VarName[var_ind]]
    for cc in range(0, len(Nlat)):
        varN[cc] = var_data[Nlat[cc], Nlon[cc]]
    return varN


##@param[in]   packdata               packaged data
##@param[in]   auxil                  auxiliary data
##@param[in]   var_ind                index of variable
##@param[in]   VarName                list of variables' names
##@param[in]   px                     index of PFT
##@retval      VarN                   variable values of selected pixels
def extract_XN(packdata, auxil, var_ind, VarName, px):
    Nlat = auxil.Nlat
    Nlon = auxil.Nlon
    varN = np.full(len(Nlat), np.nan)
    var_data = packdata[VarName[var_ind]]
    var_pft_map = np.squeeze(var_data[px - 1][:][:])
    for cc in range(0, len(Nlat)):
        varN[cc] = var_pft_map[Nlat[cc], Nlon[cc]]
    return varN


##@param[in]   packdata               packaged data
##@param[in]   auxil                  auxiliary data
##@param[in]   PFT_mask               PFT mask
##@param[in]   px                     index of PFT
##@retval      VarN                   variable values of selected pixels
def pft(packdata, auxil, PFT_mask, px):
    Nlat = auxil.Nlat
    Nlon = auxil.Nlon
    varN = np.full(len(Nlat), np.nan)
    for cc in range(0, len(Nlat)):
        varN[cc] = PFT_mask[px - 1, Nlat[cc], Nlon[cc]]
    return varN


##@param[in]   packdata               packaged data
##@param[in]   auxil                  auxiliary data
##@param[in]   ipft                   ith pft
##@retval      extr_var               extracked data
def var(packdata, auxil, ipft):
    extr_var = np.zeros(shape=(len(auxil.Nlat), 0))
    for indx in range(auxil.Nv_total):
        if indx < auxil.Nv_nopft:
            extracted_var = np.reshape(
                extract_X(packdata, auxil, indx, auxil.var_pred_name),
                (len(auxil.Nlat), 1),
            )
            extr_var = np.concatenate((extr_var, extracted_var), axis=1)
        else:
            extracted_var = np.reshape(
                extract_XN(packdata, auxil, indx, auxil.var_pred_name, ipft),
                (len(auxil.Nlat), 1),
            )
            extr_var = np.concatenate((extr_var, extracted_var), axis=1)
    return extr_var
