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


def extract_X(packdata, var_name):
    """
    Extract variable values for all pixels

    Parameters
    ----------
    packdata : xarray.Dataset
        PackData object
    var_name : str
        name of variable (e.g. Tmean, Tstd, Tmin, Tmax, etc.)

    Returns
    -------
    var : numpy.ndarray
        variable values of selected pixels
    """
    var = packdata[var_name].values
    return var[..., packdata.Nlat, packdata.Nlon]


def extract_XN(packdata, var_name, px):
    """
    Extract values for higher dimensional variables (e.g. 'NPP0', 'LAIO')
    for all grid pixels.

    Parameters
    ----------
    packdata : xarray.Dataset
        PackData object
    var_name : str
        name of variable
    px : int
        index of PFT

    Returns
    -------
    var : numpy.ndarray
        variable values of selected pixels
    """
    var = packdata[var_name].values
    return var[px - 1, packdata.Nlat, packdata.Nlon]


##@param[in]   packdata               packaged data
##@param[in]   PFT_mask               PFT mask
##@param[in]   px                     index of PFT
##@retval      VarN                   variable values of selected pixels
"""
"""


def pft(packdata, PFT_mask, px):
    return PFT_mask[px - 1, packdata.Nlat, packdata.Nlon]


def var(packdata, ipft):
    """
    Extract variable values for a given PFT. We call two different functions
    depending on the dimension of the variable.

    Parameters
    ----------
    packdata : xarray.Dataset
        PackData object
    ipft : int
        index of PFT

    Returns
    -------
    extr_var : numpy.ndarray
        extracted variable values of selected pixels
        has dimension
    """
    extr_var = []
    for var_name in packdata.data_vars:
        print(var_name)
        if "veget" not in packdata[var_name].dims:
            extracted_var = extract_X(packdata, var_name)
        else:
            extracted_var = extract_XN(packdata, var_name, ipft)
        extr_var.append(extracted_var.reshape(-1, len(packdata.Nlat), 1))

    # find the common shape
    com_shape = max(map(np.shape, extr_var))

    extr_var = [np.resize(a, com_shape) for a in extr_var]
    return np.concatenate(extr_var, axis=-1)
