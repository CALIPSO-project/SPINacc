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

def MLmap(packdata,auxil,ivar,PFT_mask,PFT_mask_lai,var_pred_name,ipool,ipft,logfile,varname,varlist,labx,ii,resultpath,fx,fy,fz,fz2,fz3,f1,f2,f3,fxx,fyy,fzz,ff1,ff2,ff3,ffz2,ffz3,loocv):
  check.display('processing %s, variable %s...'%(ipool,varname),logfile)

  # extract data
  extr_var=extract_X.var(packdata,auxil,ipft)
  # extract PFT map
  pft_ny=extract_X.pft(packdata,auxil,PFT_mask_lai,ipft).reshape(len(auxil.Nlat),1)

  # extract Y
  pool_arr=np.full(len(auxil.Nlat),np.nan)
  pool_map=np.squeeze(ivar)#[tuple(ind-1)] # all indices start from 1, but python loop starts from 0
  pool_map[pool_map==1e20]=np.nan
  for cc in range(len(auxil.Nlat)):
    pool_arr[cc]=pool_map[auxil.Nlat[cc],auxil.Nlon[cc]]
  # end extract Y 
  extracted_Y=np.reshape(pool_arr,(len(auxil.Nlat),1))
  extr_all=np.concatenate((extracted_Y,extr_var,pft_ny),axis=1)
  df_data=DataFrame(extr_all,columns=[labx])# convert the array into dataframe
  #df_data.ix[:,22]=(df_data.ix[:,22].astype(int)).astype(str)
  combine_XY=df_data.dropna()# delete pft=nan
  combine_XY=combine_XY.drop(['pft'],axis=1)
  if len(combine_XY)==0:
    check.display('%s, variable %s : NO DATA in training set!'%(ipool,varname),logfile)
    fx.write('%.2f' % np.nan+',')
    fy.write('%.2f' % np.nan+',')
    fz.write('%.2f' % np.nan+',')
    f1.write('%.2f' % np.nan+',')
    f2.write('%.2f' % np.nan+',')
    f3.write('%.2f' % np.nan+',')
    fz2.write('%.2f' % np.nan+',')
    fz3.write('%.2f' % np.nan+',')
    fxx.write('%.2f' % np.nan+',')
    fyy.write('%.2f' % np.nan+',')
    fzz.write('%.2f' % np.nan+',')
    ff1.write('%.2f' % np.nan+',')
    ff2.write('%.2f' % np.nan+',')
    ff3.write('%.2f' % np.nan+',')
    ffz2.write('%.2f' % np.nan+',')
    ffz3.write('%.2f' % np.nan+',')
    print(varname,'ahhhhhhhh')
    fx.write("\n")
    fy.write("\n")
    fz.write("\n")
    fz2.write("\n")
    fz3.write("\n")
    f1.write("\n")
    f2.write("\n")
    f3.write("\n")
    fxx.write("\n")
    fyy.write("\n")
    fzz.write("\n")
    ff1.write("\n")
    ff2.write("\n")
    ff3.write("\n")
    return
  # need Yan Sun to modify it
  if 'allname_type' in varlist['pred'].keys():
    col_type=labx.index(varlist['pred']['allname_type'])
    type_val=varlist['pred']['type_code']
    combineXY=encode.en_code(combine_XY,col_type,type_val)
  else:
    col_type='None'
    type_val='None'
    combineXY=combine_XY
  #combine_XY=pd.get_dummies(combine_XY) # one-hot encoded
  Tree_Ens,predY_train,loocv_R2,loocv_reMSE,loocv_slope,loocv_dNRMSE, loocv_sNRMSE, loocv_iNRMSE, loocv_f_SB,loocv_f_SDSD,loocv_f_LSC =train.training_BAT(combineXY,logfile,loocv)

  if not Tree_Ens:
    # only one value
    predY=np.where(pool_map==pool_map,predY_train[0],np.nan)
    Global_Predicted_Y_map=predY
  else:
    Global_Predicted_Y_map,predY=mapGlobe.extrp_global(packdata,auxil,ipft,PFT_mask,var_pred_name,\
                                                       Tree_Ens,col_type,type_val,var_pred_name)

  if (PFT_mask[ipft-1]>0).any():
    # evaluation
    R2,RMSE,slope,reMSE,dNRMSE,sNRMSE,iNRMSE,f_SB,f_SDSD,f_LSC = MLeval.evaluation_map(Global_Predicted_Y_map,pool_map,ipft,PFT_mask)
    check.display('%s, variable %s : R2=%.3f , RMSE=%.2f, slope=%.2f, reMSE=%.2f'%(ipool,varname,R2,RMSE,slope,reMSE),logfile)
    # save R2, RMSE, slope to txt files
    #fx.write('%.2f' % R2+',')
    #plot the results
    fig=plt.figure(figsize=[12,12])
    # training dat
    ax1=plt.subplot(221)
    ax1.scatter(combineXY.iloc[:,0].values,predY_train)
    # global dta
    ax2=plt.subplot(222)
#    predY=Global_Predicted_Y_map.flatten()
#    simuY=pool_map.flatten()
    ax2.scatter(pool_map[PFT_mask[ipft-1]>0],Global_Predicted_Y_map[PFT_mask[ipft-1]>0])
    xx=np.linspace(0,np.nanmax(pool_map),10)
    yy=np.linspace(0,np.nanmax(pool_map),10)
    ax2.text(0.1*np.nanmax(pool_map),0.7*np.nanmax(Global_Predicted_Y_map),"R2=%.2f"%R2)
    ax2.text(0.1*np.nanmax(pool_map),0.8*np.nanmax(Global_Predicted_Y_map),"RMSE=%i"%RMSE)
    ax2.plot(xx,yy,'k--')
    ax2.set_xlabel('full model')
    ax2.set_ylabel('Machine-learning predicted')
    ax3=plt.subplot(223)
    im=ax3.imshow(pool_map,vmin=0,vmax=0.8*np.nanmax(pool_map))
    ax3.set_title('full model')
    plt.colorbar(im,orientation='horizontal')
    ax4=plt.subplot(224)
    im=ax4.imshow(Global_Predicted_Y_map,vmin=0,vmax=0.8*np.nanmax(pool_map))
    ax4.set_title('Machine-learning predicted')
    plt.colorbar(im,orientation='horizontal')

    fig.savefig(resultpath+'Eval_%s'%varname+'.png')
    plt.close('all')
  else:
    check.display('%s, variable %s : NO DATA!'%(ipool,varname),logfile)
  fx.write('%.2f' % R2+',')
  fy.write('%.2f' % slope+',')
  fz.write('%.2f' % dNRMSE+',')
  f1.write('%.2f' % f_SB+',')
  f2.write('%.2f' % f_SDSD+',')
  f3.write('%.2f' % f_LSC+',')
  fz2.write('%.2f' % sNRMSE+',')
  fz3.write('%.2f' % iNRMSE+',')
  fxx.write('%.2f' % loocv_R2+',')
  fyy.write('%.2f' % loocv_slope+',')
  fzz.write('%.2f' % loocv_dNRMSE+',')
  ff1.write('%.2f' % loocv_f_SB+',')
  ff2.write('%.2f' % loocv_f_SDSD+',')
  ff3.write('%.2f' % loocv_f_LSC+',')
  ffz2.write('%.2f' % loocv_sNRMSE+',')
  ffz3.write('%.2f' % loocv_iNRMSE+',')
  print(varname,'ahhhhh')
  fx.write("\n")
  fy.write("\n")
  fz.write("\n")
  fz2.write("\n")
  fz3.write("\n")
  f1.write("\n")
  f2.write("\n")
  f3.write("\n")
  fxx.write("\n")
  fyy.write("\n")
  fzz.write("\n")
  ff1.write("\n")
  ff2.write("\n")
  ff3.write("\n")
  ffz2.write("\n")
  ffz3.write("\n")
  return

def MLmap_multidim(packdata,auxil,ivar,PFT_mask,PFT_mask_lai,var_pred_name,ipool,ipft,logfile,varname,varlist,labx,ind,ii,resultpath,fx,fy,fz,fz2,fz3,f1,f2,f3,fxx,fyy,fzz,ff1,ff2,ff3,ffz2,ffz3,loocv):
  check.display('processing %s, variable %s, index %s (dim: %s)...'%(ipool,varname,ind,ii['dim_loop']),logfile)

  # extract data
  extr_var=extract_X.var(packdata,auxil,ipft)
  # extract PFT map
  pft_ny=extract_X.pft(packdata,auxil,PFT_mask_lai,ipft).reshape(len(auxil.Nlat),1)

  # extract Y
  pool_arr=np.full(len(auxil.Nlat),np.nan)
  pool_map=np.squeeze(ivar)[tuple(ind-1)] # all indices start from 1, but python loop starts from 0
  pool_map[pool_map==1e20]=np.nan
  for cc in range(len(auxil.Nlat)):
    pool_arr[cc]=pool_map[auxil.Nlat[cc],auxil.Nlon[cc]]
  # end extract Y 
  extracted_Y=np.reshape(pool_arr,(len(auxil.Nlat),1))
  extr_all=np.concatenate((extracted_Y,extr_var,pft_ny),axis=1)
  df_data=DataFrame(extr_all,columns=[labx])# convert the array into dataframe
  #df_data.ix[:,22]=(df_data.ix[:,22].astype(int)).astype(str)
  combine_XY=df_data.dropna()# delete pft=nan
  combine_XY=combine_XY.drop(['pft'],axis=1)
  if len(combine_XY)==0:
    check.display('%s, variable %s, index %s (dim: %s) : NO DATA in training set!'%(ipool,varname,ind,ii['dim_loop']),logfile)
    fx.write('%.2f' % np.nan+',')
    fy.write('%.2f' % np.nan+',')
    fz.write('%.2f' % np.nan+',')
    f1.write('%.2f' % np.nan+',')
    f2.write('%.2f' % np.nan+',')
    f3.write('%.2f' % np.nan+',')
    fz2.write('%.2f' % np.nan+',')
    fz3.write('%.2f' % np.nan+',')
    fxx.write('%.2f' % np.nan+',')
    fyy.write('%.2f' % np.nan+',')
    fzz.write('%.2f' % np.nan+',')
    ff1.write('%.2f' % np.nan+',')
    ff2.write('%.2f' % np.nan+',')
    ff3.write('%.2f' % np.nan+',')
    ffz2.write('%.2f' % np.nan+',')
    ffz3.write('%.2f' % np.nan+',')
    if ind[-1]==ii['loops'][ii['dim_loop'][-1]][-1]:
      print(varname,ind)
      fx.write("\n")
      fy.write("\n")
      fz.write("\n")
      fz2.write("\n")
      fz3.write("\n")
      f1.write("\n")
      f2.write("\n")
      f3.write("\n")
      fxx.write("\n")
      fyy.write("\n")
      fzz.write("\n")
      ff1.write("\n")
      ff2.write("\n")
      ff3.write("\n")
    return
  # need Yan Sun to modify it
  if 'allname_type' in varlist['pred'].keys():
    col_type=labx.index(varlist['pred']['allname_type'])
    type_val=varlist['pred']['type_code']
    combineXY=encode.en_code(combine_XY,col_type,type_val)
  else:
    col_type='None'
    type_val='None'
    combineXY=combine_XY
  #combine_XY=pd.get_dummies(combine_XY) # one-hot encoded
  Tree_Ens,predY_train,loocv_R2,loocv_reMSE,loocv_slope,loocv_dNRMSE, loocv_sNRMSE, loocv_iNRMSE, loocv_f_SB,loocv_f_SDSD,loocv_f_LSC =train.training_BAT(combineXY,logfile,loocv)

  if not Tree_Ens:
    # only one value
    predY=np.where(pool_map==pool_map,predY_train[0],np.nan)
    Global_Predicted_Y_map=predY
  else:
    Global_Predicted_Y_map,predY=mapGlobe.extrp_global(packdata,auxil,ipft,PFT_mask,var_pred_name,\
                                                       Tree_Ens,col_type,type_val,var_pred_name)

  if (PFT_mask[ipft-1]>0).any():
    # evaluation
    R2,RMSE,slope,reMSE,dNRMSE,sNRMSE,iNRMSE,f_SB,f_SDSD,f_LSC = MLeval.evaluation_map(Global_Predicted_Y_map,pool_map,ipft,PFT_mask)
    check.display('%s, variable %s, index %s (dim: %s) : R2=%.3f , RMSE=%.2f, slope=%.2f, reMSE=%.2f'%(ipool,varname,ind,ii['dim_loop'],R2,RMSE,slope,reMSE),logfile)
    # save R2, RMSE, slope to txt files
    #fx.write('%.2f' % R2+',')
    #plot the results
    fig=plt.figure(figsize=[12,12])
    # training dat
    ax1=plt.subplot(221)
    ax1.scatter(combineXY.iloc[:,0].values,predY_train)
    # global dta
    ax2=plt.subplot(222)
#    predY=Global_Predicted_Y_map.flatten()
#    simuY=pool_map.flatten()
    ax2.scatter(pool_map[PFT_mask[ipft-1]>0],Global_Predicted_Y_map[PFT_mask[ipft-1]>0])
    xx=np.linspace(0,np.nanmax(pool_map),10)
    yy=np.linspace(0,np.nanmax(pool_map),10)
    ax2.text(0.1*np.nanmax(pool_map),0.7*np.nanmax(Global_Predicted_Y_map),"R2=%.2f"%R2)
    ax2.text(0.1*np.nanmax(pool_map),0.8*np.nanmax(Global_Predicted_Y_map),"RMSE=%i"%RMSE)
    ax2.plot(xx,yy,'k--')
    ax2.set_xlabel('full model')
    ax2.set_ylabel('Machine-learning predicted')
    ax3=plt.subplot(223)
    im=ax3.imshow(pool_map,vmin=0,vmax=0.8*np.nanmax(pool_map))
    ax3.set_title('full model')
    plt.colorbar(im,orientation='horizontal')
    ax4=plt.subplot(224)
    im=ax4.imshow(Global_Predicted_Y_map,vmin=0,vmax=0.8*np.nanmax(pool_map))
    ax4.set_title('Machine-learning predicted')
    plt.colorbar(im,orientation='horizontal')

    fig.savefig(resultpath+'Eval_%s'%varname+''.join(['_'+ii['dim_loop'][ll]+'%2.2i'%ind[ll] for ll in range(len(ind))]+['.png']))
    plt.close('all')
  else:
    check.display('%s, variable %s, index %s (dim: %s) : NO DATA!'%(ipool,varname,ind,ii['dim_loop']),logfile)
  fx.write('%.2f' % R2+',')
  fy.write('%.2f' % slope+',')
  fz.write('%.2f' % dNRMSE+',')
  f1.write('%.2f' % f_SB+',')
  f2.write('%.2f' % f_SDSD+',')
  f3.write('%.2f' % f_LSC+',')
  fz2.write('%.2f' % sNRMSE+',')
  fz3.write('%.2f' % iNRMSE+',')
  fxx.write('%.2f' % loocv_R2+',')
  fyy.write('%.2f' % loocv_slope+',')
  fzz.write('%.2f' % loocv_dNRMSE+',')
  ff1.write('%.2f' % loocv_f_SB+',')
  ff2.write('%.2f' % loocv_f_SDSD+',')
  ff3.write('%.2f' % loocv_f_LSC+',')
  ffz2.write('%.2f' % loocv_sNRMSE+',')
  ffz3.write('%.2f' % loocv_iNRMSE+',')
  if ind[-1]==ii['loops'][ii['dim_loop'][-1]][-1]:
    print(varname,ind)
    fx.write("\n")
    fy.write("\n")
    fz.write("\n")
    fz2.write("\n")
    fz3.write("\n")
    f1.write("\n")
    f2.write("\n")
    f3.write("\n")
    fxx.write("\n")
    fyy.write("\n")
    fzz.write("\n")
    ff1.write("\n")
    ff2.write("\n")
    ff3.write("\n")
    ffz2.write("\n")
    ffz3.write("\n")
  return

##@param[in]   packdata               packaged data
##@param[in]   auxil                  auxiliary data
##@param[in]   ipool                  'som','biomass' or 'litter'
##@param[in]   logfile                logfile
##@
##@param[in]
def MLloop(packdata,auxil,ipool,logfile,varlist,labx,resultpath,fx,fy,fz,fz2,fz3,f1,f2,f3,fxx,fyy,fzz,ff1,ff2,ff3,ffz2,ffz3,loocv):
  var_pred_name1=varlist['pred']['allname']
  var_pred_name2=varlist['pred']['allname_pft']
  var_pred_name=var_pred_name1+var_pred_name2

  responseY=Dataset(varlist['resp']['sourcefile'],'r')
  PFT_mask,PFT_mask_lai=genMask.PFT(packdata,varlist,varlist['PFTmask']['pred_thres'])

  Yvar=varlist['resp']['variables'][ipool]
  for ii in Yvar:
    for jj in ii['name_prefix']:
      for kk in ii['loops'][ii['name_loop']]:
        varname=jj+('_%2.2i'%kk if kk else '')+ii['name_postfix']
        if ii['name_loop']=='pft':ipft=kk
#        print(responseY.variables.keys())
        ivar=responseY[varname]
        if ii['dim_loop']==['null']:
          figname=resultpath+'Eval_%s'%varname+'.png'
#          if os.path.isfile(figname):continue
          if ipft in ii['skip_loop']['pft']:continue
          MLmap(packdata,auxil,ivar,PFT_mask,PFT_mask_lai,var_pred_name,ipool,ipft,logfile,varname,varlist,labx,ii,resultpath,fx,fy,fz,fz2,fz3,f1,f2,f3,fxx,fyy,fzz,ff1,ff2,ff3,ffz2,ffz3,loocv)
        else:
#          index=np.array(np.meshgrid(*[ii['loops'][ll] for ll in ii['dim_loop']])).T.reshape(-1,len(ii['dim_loop']))
          index=np.array(list(itertools.product(*[ii['loops'][ll] for ll in ii['dim_loop']])))
#          print(index);sys.exit()
          for ind in index:
            figname=resultpath+'Eval_%s'%varname+''.join(['_'+ii['dim_loop'][ll]+'%2.2i'%ind[ll] for ll in range(len(ind))]+['.png'])
#            if os.path.isfile(figname):continue
            if 'pft' in ii['dim_loop']:ipft=ind[ii['dim_loop'].index('pft')]
            if ipft in ii['skip_loop']['pft']:continue
            MLmap_multidim(packdata,auxil,ivar,PFT_mask,PFT_mask_lai,var_pred_name,ipool,ipft,logfile,varname,varlist,labx,ind,ii,resultpath,fx,fy,fz,fz2,fz3,f1,f2,f3,fxx,fyy,fzz,ff1,ff2,ff3,ffz2,ffz3,loocv)
  return

