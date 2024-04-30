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
##@param[in]   PFT_mask               PFT mask where PFT fraction >0.01
##@param[in]   ipft                   ith PFT to deal with
##@param[in]   var_pred               predicting variables
##@param[in]   var_pred_name          names of predicting variables
##@param[in]   K                      K
##@param[in]   Nc                     number of sites of select
##@retval      cluster_dic            # to be complete by Yan
##@retval      distance               # to be complete by Yan
##@retval      All_selectedID         # to be complete by Yan
def Cluster_Ana(packdata, PFT_mask, ipft, var_pred_name, K, Nc):
    if "Ndep_nhx_pft" in var_pred_name:
        packdata.Ndep_nhx_pft = packdata.Ndep_nhx[ipft - 1]
    if "Ndep_noy_pft" in var_pred_name:
        packdata.Ndep_noy_pft = packdata.Ndep_noy[ipft - 1]
    if "Pdep_pft" in var_pred_name:
        packdata.Pdep_pft = packdata.Pdep[ipft - 1]
    pp = PFT_mask[ipft - 1]
    laix = packdata.LAI0[ipft - 1]
    pp[laix < 0.01] = np.nan
    var_pred = packdata[var_pred_name] * pp
    df = var_pred.to_dataframe().dropna()
    mod = KMeans(n_clusters=K)
    CC = mod.fit_predict(df)
    distance = mod.inertia_
    Cluster_output = Series(CC, index=df.index)

    cluster_dic = {}
    All_selectedID = np.empty((0, 2))
    for clus in range(K):
        A = Cluster_output[Cluster_output == clus]
        locations = np.array(A.index.to_list())
        cluster_dic["clus_%.2i_loc" % clus] = locations
        # 1.3 Randomly select Nc sites from each cluster
        if len(locations) > Nc:
            RandomS = random.sample(range(len(locations)), Nc)
            SelectedID = locations[RandomS]
        else:
            SelectedID = locations
        cluster_dic["clus_%.2i_loc_select" % clus] = SelectedID
        All_selectedID = np.append(All_selectedID, SelectedID, axis=0)

    return cluster_dic, distance, All_selectedID


##@param[in]   packdata               packaged data
##@param[in]   varlist                list of variables, including name of source files, variable names, etc.
##@param[in]   logfile                logfile
##@retval      dis_all                # Eulerian (?) distance corresponding to different number of Ks
def Cluster_test(packdata, varlist, logfile):
    # 1.clustering def
    # Make a mask map according to PFT fractions: nan - <0.00000001; 1 - >=0.00000001
    # I used the output 'VEGET_COV_MAX' by ORCHIDEE-CNP with run the spin-up for 1 year.
    # Please provide the file path and name for PFT fractions with resolution of 2 deg.
    PFT_mask, PFT_mask_lai = genMask.PFT(
        packdata, varlist, varlist["PFTmask"]["cluster_thres"]
    )

    # predictor metrcis
    var_pred_name = varlist["pred"]["clustering"]

    # 2. Use different K value and valuate the clustering results
    kvalues = list(packdata.Ks)
    kpfts = varlist["clustering"]["pfts"]
    dis_all = np.zeros(shape=(len(kvalues), len(kpfts)))
    for veg in kpfts:
        for kkk in kvalues:
            ClusD, disx, traID = Cluster_Ana(
                packdata, PFT_mask, veg, var_pred_name, kkk, 10
            )
            dis_all[kvalues.index(kkk), kpfts.index(veg)] = disx

    return dis_all


##@param[in]   packdata               packaged data
##@param[in]   varlist                list of variables, including name of source files, variable names, etc.
##@param[in]   KK                     K value chosen to do final clustering
##@param[in]   logfile                logfile
##@retval      IDx                    chosen IDs of pixels for MLacc
##@retval      IDloc                  # to be complete by Yan (just for plotting)
##@retval      IDsel                  # to be complete by Yan (just for plotting)
def Cluster_all(packdata, varlist, KK, logfile):
    adict = locals()
    kpfts = varlist["clustering"]["pfts"]
    Ncc = varlist["clustering"]["Ncc"]
    PFT_mask, PFT_mask_lai = genMask.PFT(
        packdata, varlist, varlist["PFTmask"]["cluster_thres"]
    )

    var_pred_name = varlist["pred"]["clustering"]
    for veg in kpfts:
        ClusD, disx, training_ID = Cluster_Ana(
            packdata,
            PFT_mask,
            veg,
            var_pred_name,
            KK,
            Ncc[kpfts.index(veg)],
        )
        #    locations=ClusD['Aloc']
        #    SelectedID=ClusD['Aloc_select']
        adict["PFT" + str(veg) + "ClusD"] = ClusD
        adict["PFT" + str(veg) + "trainingID"] = training_ID

    # 4. Check if the selected sites are representative? (not sure)

    # 5. Output the ID
    IDx = np.concatenate([adict["PFT%itrainingID" % ii] for ii in kpfts])
    IDx = np.unique(IDx, axis=0)
    IDloc = np.array([adict["PFT%iClusD" % ii]["clus_01_loc"] for ii in kpfts])
    IDsel = np.array([adict["PFT%iClusD" % ii]["clus_01_loc_select"] for ii in kpfts])
    return IDx, IDloc, IDsel
