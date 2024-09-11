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
    Extract variable values for selected pixels.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        var_name (str): Name of the variable to extract.

    Returns:
        numpy.ndarray: Variable values for selected pixels.
    """
    var = packdata[var_name].values
    return var[..., packdata.Nlat, packdata.Nlon]


def extract_XN(packdata, var_name, px):
    """
    Extract variable values for selected pixels for a specific PFT.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        var_name (str): Name of the variable to extract.
        px (int): Index of the PFT.

    Returns:
        numpy.ndarray: Variable values for selected pixels and PFT.
    """
    var = packdata[var_name].values
    return var[px - 1, packdata.Nlat, packdata.Nlon]


def pft(packdata, PFT_mask, px):
    """
    Extract PFT mask for selected pixels.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        PFT_mask (numpy.ndarray): Mask for Plant Functional Types.
        px (int): Index of the PFT.

    Returns:
        numpy.ndarray: PFT mask values for selected pixels.
    """
    return PFT_mask[px - 1, packdata.Nlat, packdata.Nlon]


def var(packdata, ipft):
    """
    Extract all variables for a specific PFT.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        ipft (int): Index of the PFT.

    Returns:
        numpy.ndarray: Array of extracted variable values for the specified PFT.
    """
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
