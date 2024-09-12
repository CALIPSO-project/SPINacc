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


def PFT(packdata, varlist, thres):
    """
    Generate Plant Functional Type (PFT) masks based on PFT fraction and LAI.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables, including LAI.
        varlist (dict): Dictionary of variable information, including PFT mask source file and variable name.
        thres (float): Threshold for PFT fraction to be defined as a valid pixel.

    Returns:
        tuple:
            - PFT_mask (numpy.ndarray): Mask where PFT fraction > threshold (1 for valid, NaN for invalid).
            - PFT_mask_lai (numpy.ndarray): Mask where PFT fraction > threshold and LAI >= 0 (1 for valid, NaN for invalid).
    """
    f = Dataset(varlist["PFTmask"]["sourcefile"], "r")
    mkk = f[varlist["PFTmask"]["var"]][-1].filled(np.nan)
    PFT_fraction = np.squeeze(mkk)
    PFT_mask = np.full(PFT_fraction.shape, np.nan)
    #  if not np.isnan(PFT_fraction[0][0][0]):
    #    PFT_fraction[PFT_fraction==PFT_fraction[0][0][0]]=np.nan
    PFT_mask[PFT_fraction > thres] = 1
    PFT_mask_lai = np.where(packdata.LAI0 < 0, np.nan, PFT_mask)
    return PFT_mask, PFT_mask_lai
