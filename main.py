#!/usr/bin/env python
##### -*- coding: utf-8 -*-
"""
 MLacc - Machine-Learning-based acceleration of spin-up

 Copyright Laboratoire des Sciences du Climat et de l'Environnement (LSCE)
           Unite mixte CEA-CNRS-UVSQ
 
 Code manager:
 Daniel Goll, LSCE, dsgoll123@gmail.com
 
 This software is developed by Yan Sun, Yilong Wang and Daniel Goll.......
 
 This software is governed by the XXX license
 XXXX <License content>
"""

from Tools import *

# print Python version 
print(sys.version)

#
# Read configuration file
#

if len( sys.argv ) < 2:
  print('Missing argument: DEF directory')
  sys.exit()
else:
  dir_def = sys.argv[1]

f=open(dir_def+'MLacc.def','r')
config=f.readlines()
f.close()

# Define standard output file 
thefile=config[1].strip()

# Define task 
itask=config[3].strip().split(',')

logfile=open(thefile,'w',1)
check.display('DEF directory: '+dir_def,logfile)
check.display('running task: %s'%str(itask),logfile)

# Define task 
resultpath=config[5].strip()+'/'
check.display('results are stored at: '+resultpath,logfile)

# Read list of variables
varfile=open(dir_def+'varlist.json','r')
varlist=json.loads(varfile.read())

# load stored results or start from scratch
iprec=int(config[7].strip())
if iprec:
  check.display('read from previous results...',logfile)
  packdata=np.load(resultpath+'packdata.npy',allow_pickle=True).item()
  auxil=np.load(resultpath+'auxil.npy',allow_pickle=True).item()
else:
  check.display('MLacc start from scratch...',logfile)
  # initialize packaged data and auxiliary data
  packdata=pack()
  auxil=auxiliary()
  readvar(packdata,auxil,varlist,config,logfile)
  np.save(resultpath+'packdata.npy',packdata)
  np.save(resultpath+'auxil.npy',auxil)

# range of Ks to be tested, and the final K
maxK=int(config[11].strip())
auxil.Ks=range(2,maxK+1)
auxil.K=int(config[9].strip())  
  
# Define random seed
iseed=int(config[13].strip())
random.seed(iseed)
check.display('random seed = %i'%iseed,logfile)

if '1' in itask:
  #
  # test clustering
  dis_all=Cluster.Cluster_test(packdata,auxil,varlist,logfile) 
  dis_all.dump(resultpath+'dist_all.npy')
  check.display('test clustering done!\nResults have been stored as dist_all.npy',logfile)

  #
  # plot clustering results
  fig,ax=plt.subplots()
  lns=[]
  for ipft in range(dis_all.shape[1]):
    lns+=ax.plot(auxil.Ks,dis_all[:,ipft])
  plt.legend(lns,varlist['clustering']['pfts'],title="PFT")
  ax.set_ylabel('Sum of squared distances of samples to\ntheir closest cluster center')
  ax.set_xlabel('K-value (cluster size)')
  fig.savefig(resultpath+'dist_all.png') 
  plt.close('all')
  check.display('test clustering results plotted!\nResults have been stored as dist_all.png',logfile)

if '2' in itask:
  #
  # clustering
  KK=int(config[9].strip())
  check.display('Kmean algorithm, K=%i'%KK,logfile)
  IDx,IDloc,IDsel=Cluster.Cluster_all(packdata,auxil,varlist,KK,logfile)
  IDx.dump(resultpath+'IDx.npy')
  np.savetxt(resultpath+'IDx.txt',IDx,fmt='%.2f')
  IDloc.dump(resultpath+'IDloc.npy')
  IDsel.dump(resultpath+'IDsel.npy')
  check.display('clustering done!\nResults have been stored as IDx.npy',logfile)

  #
  # plot clustering results
  kpfts=varlist['clustering']['pfts']
  for ipft in range(len(kpfts)):
    fig,ax=plt.subplots()
    m=Basemap()
    m.drawcoastlines()
    m.scatter(IDloc[ipft][:,1],IDloc[ipft][:,0],s=10,marker='o',c='gray')
    m.scatter(IDsel[ipft][:,1],IDsel[ipft][:,0],s=10,marker='o',c='red')
    fig.savefig(resultpath+'ClustRes_PFT%i.png'%kpfts[ipft])
    plt.close('all')
  check.display('clustering results plotted!\nResults have been stored as ClustRes_PFT*.png',logfile)

if '3' in itask:
  #
  # ML extrapolation

  adict=locals()
  var_pred_name1=varlist['pred']['allname']
  var_pred_name2=varlist['pred']['allname_pft']
  var_pred_name=var_pred_name1+var_pred_name2
  auxil.Nv_nopft=len(var_pred_name1)
  auxil.Nv_total=len(var_pred_name)
  auxil.var_pred_name=var_pred_name
  
  # Response variables
  Yvar=varlist['resp']['variables']
  for var_key in Yvar.keys():
    ilist=Yvar[var_key] 
    resp_list=[]
    for ii in ilist:
      resp_list+=[ii['name_prefix']+'_%2.2i'%jj+ii['name_postfix'] for jj in ii['pfts']]
    adict['var_resp_%s'%var_key]=resp_list
  responseY=Dataset(varlist['resp']['sourcefile'],'r')
  
  check.check_file(resultpath+'IDx.npy',logfile)
  IDx=np.load(resultpath+'IDx.npy',allow_pickle=True)
  # generate PFT mask
  PFT_mask,PFT_mask_lai=genMask.PFT(packdata,varlist,varlist['PFTmask']['pred_thres'],logfile)

  auxil.Nlat=(np.trunc((90-IDx[:,0])/auxil.lat_reso)).astype(int)
  auxil.Nlon=np.trunc((180+IDx[:,1])/auxil.lon_reso).astype(int)
  labx=['Y']+var_pred_name+['pft']
  for ipool in Yvar.keys():
#    if ipool=='som':continue
    check.display('processing %s...'%ipool,logfile)
    auxil.pfts=Yvar[ipool][0]['pfts']
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
          exec('pool_arr,pool_map=extract_Y.extract(responseY,auxil,var_resp_%s[indy],isubp)'%(ipool))
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
            R2,RMSE,slope=MLeval.evaluation_map(Global_Predicted_Y_map,pool_map,auxil.pfts[ipft],PFT_mask)
            check.display('%s PFT%2.2i element%i subp%i: R2=%.3f , RMSE=%.2f, slope=%.2f'%(ipool,auxil.pfts[ipft],cnp.index(indy),isubp,R2,RMSE,slope),logfile)

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
  check.display('task 3 done!',logfile)
          
  
  
