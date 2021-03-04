#=============================================================================================
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
#=============================================================================================

from Tools import *

##@param[in]   responseY              netcdf file containing target variables
##@param[in]   auxil                  auxiliary data
##@param[in]   VarName                name of variable to extract
##@param[in]   isubp                  ith sub-pool                            
##@retval      VarN                   variable values of selected pixels
##@retval      var_da[isubp]          global map of ith sub-pool
def extract(responseY,auxil,VarName,isubp):
    varN=np.full(len(auxil.Nlat),np.nan)
    var_da=np.squeeze(responseY[VarName])
    var_da[var_da==1e20]=np.nan
    for cc in range(len(auxil.Nlat)):
        varN[cc]=var_da[isubp,auxil.Nlat[cc],auxil.Nlon[cc]]
    return varN, var_da[isubp]

