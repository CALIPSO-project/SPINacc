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

##@param[in]   ipft                   index of PFT 
##@param[in]   PFTmask                PFT mask
##@param[in]   XVarName               input variables
##@param[in]   Tree_Ens               tree ensemble
##@param[in]   colum                  
##@param[in]   Nm                     
##@param[in]   labx                   
##@retval      R2                     R2 between predicted Y and target Y
##@retval      RMSE                   RMSE of prediceted Y
##@retval      slope                  regression slope of target Y and predicted Y
def evaluation_map(Global_Predicted_Y_map,Y_map,ipft,PFTmask):
  pmask=np.squeeze(PFTmask[ipft-1][:])
  all_predY=np.reshape(Global_Predicted_Y_map,(-1,1))
  all_Y=np.reshape(Y_map*pmask,(-1,1))
  #all_Y[all_Y<0.000000001]=np.nan
  allyy=DataFrame(np.concatenate((all_Y, all_predY),axis=1))
  allyy=allyy.dropna()
  comp_Y=allyy.values
  MSE=mean_squared_error(comp_Y[:,0],comp_Y[:,1])
  RMSE=np.sqrt(mean_squared_error(comp_Y[:,0],comp_Y[:,1]))
  R2=r2_score(comp_Y[:,0],comp_Y[:,1])
  slope, intercept, r_value, p_value, std_err = stats.linregress(comp_Y[:,0],comp_Y[:,1])

  return R2, RMSE, slope
