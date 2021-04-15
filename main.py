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
  responseY=Dataset(varlist['resp']['sourcefile'],'r')
  
  check.check_file(resultpath+'IDx.npy',logfile)
  IDx=np.load(resultpath+'IDx.npy',allow_pickle=True)
  # generate PFT mask
  PFT_mask,PFT_mask_lai=genMask.PFT(packdata,varlist,varlist['PFTmask']['pred_thres'],logfile)

  auxil.Nlat=(np.trunc((90-IDx[:,0])/auxil.lat_reso)).astype(int)
  auxil.Nlon=np.trunc((180+IDx[:,1])/auxil.lon_reso).astype(int)
  labx=['Y']+var_pred_name+['pft']
  for ipool in Yvar.keys():
    check.display('processing %s...'%ipool,logfile)
    fx=open(resultpath+ipool+'_R2.txt','w')
    fy=open(resultpath+ipool+'_slope.txt','w')
    fz=open(resultpath+ipool+'_reMSE.txt','w')
    if ipool!='biomass':
      auxil.pfts=Yvar[ipool][0]['pfts']
      print(auxil.pfts)
      ML_som_litter.MLloop(packdata,auxil,ipool,logfile,varlist,labx,resultpath,fx,fy,fz)
    else:
      auxil.pfts=range(2,varlist['npfts']+2)#[1:]#Yvar[ipool][0]['pfts']
      print(auxil.pfts)
      ML_biomass.MLloop(packdata,auxil,ipool,logfile,varlist,labx,resultpath,fx,fy,fz)
    fx.close()
    fy.close()
    fz.close()     
  check.display('task 3 done!',logfile)
if '4' in itask:
  Yvar=varlist['resp']['variables']
  for ipool in Yvar.keys():
    if ipool!="biomass":continue
    subpool_name=varlist['resp']['pool_name_'+ipool]
    #check.display(subpool_name,logfile)
    npfts=varlist['npfts']
    n_cnp=varlist['cnp']
    eval_plot.plot_metric(resultpath,npfts,ipool,n_cnp,subpool_name)
  check.display('task 4 done!',logfile)
  
