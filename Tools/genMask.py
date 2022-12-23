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

##@param[in]   packdata               packaged data
##@param[in]   varlist                list of variables, including name of source files, variable names, etc.
##@param[in]   thres                  threshold for PFT fraction to be defined as valid pixel
##@retval      PFT_mask               PFT mask where PFT fraction < thres
##@retval      PFT_mask_lai           PFT mask where LAI <0
def PFT(packdata,varlist,thres):
  f=Dataset(varlist['PFTmask']['sourcefile'],'r')
  mkk=f[varlist['PFTmask']['var']][-1].filled(np.nan)
  PFT_fraction=np.squeeze(mkk)
  PFT_mask=np.full(PFT_fraction.shape,np.nan)
#  if not np.isnan(PFT_fraction[0][0][0]):
#    PFT_fraction[PFT_fraction==PFT_fraction[0][0][0]]=np.nan
  PFT_mask[PFT_fraction>thres]=1
  PFT_mask_lai=np.where(packdata.LAI0<0,np.nan,PFT_mask)
  return PFT_mask,PFT_mask_lai

