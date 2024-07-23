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
##@param[in]   ipft                   index of PFT
##@param[in]   PFTmask                PFT mask
##@param[in]   XVarName               input variables
##@param[in]   Tree_Ens               tree ensemble
##@param[in]   colum
##@param[in]   Nm
##@param[in]   labx
##@retval      Pred_Y_map             predicted map of target variables, masking nan pixels
##@retval      Pred_Y                 predicted map of target variables, without masking
def extrp_global(packdata, PFTmask, XVarName, Tree_Ens, colum, Nm, labx):
    global_X_map = np.full((len(XVarName), packdata.nlat, packdata.nlon), np.nan)
    # PFTmask[np.isnan(PFTmask)]=0
    Pred_Y = np.full(PFTmask[0].shape, np.nan)
    # global metrics -> dataframe
    for ii in range(len(XVarName)):
        if ii < packdata.Nv_nopft:
            global_X_map[ii] = packdata[XVarName[ii]][:]
        else:
            global_X_map[ii] = np.squeeze(packdata[XVarName[ii]])
        #    global_X_map=lc['global_X_map']
        das = global_X_map.transpose(1, 2, 0)
    for llat in range(packdata.nlat):
        Xllat = das[llat]
        # Xllat[np.isnan(Xllat)]=-9999
        Xtr = DataFrame(Xllat, columns=[labx])
        ind = Xtr.index
        Xtr = Xtr.dropna()
        if len(Xtr) > 0:
            if colum != "None":
                Xtr_encode = encode.en_code(Xtr, colum - 1, Nm)
            else:
                Xtr_encode = Xtr
            # Xtr.ix[:,colum]=(Xtr.ix[:,colum].astype(int)).astype(str)
            # Xtrr=pd.get_dummies(Xtr)
            Ym = DataFrame(Tree_Ens.predict(Xtr_encode))
            Ym.index = Xtr_encode.index
            Ymm = Ym.reindex(index=range(max(ind) + 1))
            Pred_Y[llat][:] = np.squeeze(Ymm)

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
