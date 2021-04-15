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
##@param[in]   auxil                  auxiliary data
##@param[in]   ipool                  'som','biomass' or 'litter'
##@param[in]   logfile                logfile
##@
##@param[in]

def MLloop(packdata,auxil,ipool,logfile,varlist,labx,resultpath,fx,fy,fz):
  adict=locals()
  var_pred_name1=varlist['pred']['allname']
  var_pred_name2=varlist['pred']['allname_pft']
  var_pred_name=var_pred_name1+var_pred_name2
  Yvar=varlist['resp']['variables']
  for var_key in Yvar.keys():
    ilist=Yvar[var_key]
    resp_list=[]
    for ii in ilist:
      resp_list+=[ii['name_prefix']+'_%2.2i'%jj+ii['name_postfix'] for jj in ii['pfts']]
    adict['var_resp_%s'%var_key]=resp_list
  responseY=Dataset(varlist['resp']['sourcefile'],'r')
  PFT_mask,PFT_mask_lai=genMask.PFT(packdata,varlist,varlist['PFTmask']['pred_thres'],logfile)
  for ipft in range(len(auxil.pfts)):
    check.display('processing %s, PFT %2.2i...'%(ipool,auxil.pfts[ipft]),logfile)
    # extract data
    extr_var=extract_X.var(packdata,auxil,auxil.pfts[ipft])
    # extract PFT map
    pft_ny=extract_X.pft(packdata,auxil,PFT_mask_lai,auxil.pfts[ipft]).reshape(len(auxil.Nlat),1)
    # index of C, N and P pools
    if len(adict['var_resp_%s'%ipool])==len(auxil.pfts):
      cnp=[ipft,]
    elif len(adict['var_resp_%s'%ipool])==len(auxil.pfts)*2:
      cnp=[ipft,ipft+len(auxil.pfts)] # SOC,N
    else:
      cnp=[ipft,ipft+len(auxil.pfts),ipft+len(auxil.pfts)*2] # SOC,N and P
    for indy in cnp:
      check.display('processing %s, PFT %2.2i, CNP %2.2i...'%(ipool,auxil.pfts[ipft],cnp.index(indy)),logfile)
      nsubp=responseY[adict['var_resp_%s'%ipool][indy]].shape[1]
      check.display('%s, PFT %2.2i, CNP %2.2i, has %2.2i subpools:'%(ipool,auxil.pfts[ipft],cnp.index(indy),nsubp),logfile)
      for isubp in range(nsubp): # loop for 4 soil pools (surface,labile,passive,slow)
        check.display('processing %s, PFT %2.2i, CNP %2.2i, subpool %2.2i...'%(ipool,auxil.pfts[ipft],cnp.index(indy),isubp),logfile)
        lc=locals()
        exec('pool_arr,pool_map=extract_Y.extract(responseY,auxil,var_resp_%s[indy],isubp)'%(ipool))
        pool_arr=lc['pool_arr']
        pool_map=lc['pool_map']
        extracted_Y=np.reshape(pool_arr,(len(auxil.Nlat),1))
        extr_all=np.concatenate((extracted_Y,extr_var,pft_ny),axis=1)
        df_data=DataFrame(extr_all,columns=[labx])# convert the array into dataframe
        #df_data.ix[:,22]=(df_data.ix[:,22].astype(int)).astype(str)
        combine_XY=df_data.dropna()# delete pft=nan
        combine_XY=combine_XY.drop(['pft'],axis=1)
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
        Tree_Ens,predY_train=train.training_BAT(combineXY,logfile)

        if not Tree_Ens:
          # only one value
          predY=np.where(pool_map==pool_map,predY_train[0],np.nan)
          Global_Predicted_Y_map=predY
        else:
          Global_Predicted_Y_map,predY=mapGlobe.extrp_global(packdata,auxil,auxil.pfts[ipft],PFT_mask,var_pred_name,\
                                                             Tree_Ens,col_type,type_val,var_pred_name)

        if (PFT_mask[auxil.pfts[ipft]-1]>0).any():
          # evaluation
          R2,RMSE,slope,reMSE=MLeval.evaluation_map(Global_Predicted_Y_map,pool_map,auxil.pfts[ipft],PFT_mask)
          check.display('%s PFT%2.2i element%i subp%i: R2=%.3f , RMSE=%.2f, slope=%.2f, reMSE=%.2f'%(ipool,auxil.pfts[ipft],cnp.index(indy),isubp,R2,RMSE,slope,reMSE),logfile)
          # save R2, RMSE, slope to txt files
          #fx.write('%.2f' % R2+',')
          #plot the results
          fig=plt.figure(figsize=[12,12])
          # training dat
          ax1=plt.subplot(221)
          ax1.scatter(combineXY.iloc[:,0].values,predY_train)
          # global dta
          ax2=plt.subplot(222)
  #          predY=Global_Predicted_Y_map.flatten()
  #          simuY=pool_map.flatten()
          ax2.scatter(pool_map[PFT_mask[auxil.pfts[ipft]-1]>0],Global_Predicted_Y_map[PFT_mask[auxil.pfts[ipft]-1]>0])
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

          fig.savefig(resultpath+'Eval_%s_PFT%i_ele%i_subp%i.png'%(ipool,auxil.pfts[ipft],cnp.index(indy),isubp))
          plt.close('all')
        else:
          check.display('%s PFT%2.2i element%i subp%i: NO DATA!'%(ipool,auxil.pfts[ipft],cnp.index(indy),isubp,R2,RMSE,slope),logfile)
        fx.write('%.2f' % R2+',')
        fy.write('%.2f' % slope+',')
        fz.write('%.2f' % reMSE+',')
      fx.write("\n")
      fy.write("\n")
      fz.write("\n")
  return
