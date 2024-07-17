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
##@param[in]   var_name               name of variable
##@retval      VarN                   variable values of selected pixels
def extract_X(packdata, var_name):
    var = packdata[var_name].values
    return var[..., packdata.Nlat, packdata.Nlon]


##@param[in]   packdata               packaged data
##@param[in]   var_name               name of variable
##@param[in]   px                     index of PFT
##@retval      VarN                   variable values of selected pixels
def extract_XN(packdata, var_name, px):
    var = packdata[var_name].values
    return var[px - 1, packdata.Nlat, packdata.Nlon]


##@param[in]   packdata               packaged data
##@param[in]   PFT_mask               PFT mask
##@param[in]   px                     index of PFT
##@retval      VarN                   variable values of selected pixels
def pft(packdata, PFT_mask, px):
    return PFT_mask[px - 1, packdata.Nlat, packdata.Nlon]


##@param[in]   packdata               packaged data
##@param[in]   ipft                   ith pft
##@retval      extr_var               extracked data
def var(packdata, ipft):
    extr_var = []
    for var_name in packdata.data_vars:
        if "veget" not in packdata[var_name].dims:
            extracted_var = extract_X(packdata, var_name)
        else:
            extracted_var = extract_XN(packdata, var_name, ipft)
        extr_var.append(extracted_var.reshape(-1, len(packdata.Nlat), 1))
    com_shape = max(map(np.shape, extr_var))
    extr_var = [np.resize(a, com_shape) for a in extr_var]
    return np.concatenate(extr_var, axis=-1)