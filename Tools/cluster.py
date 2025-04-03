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


def cluster_ana(
    packdata,
    PFT_mask,
    ipft,
    var_pred_name,
    K,
    Nc,
    adict,
    sel_most_PFTs,
    old_cluster,
    seed=1000,
):
    """
    Perform clustering analysis on the data for a specific Plant Functional Type (PFT).

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        PFT_mask (numpy.ndarray): Mask for Plant Functional Types.
        ipft (int): Index of the current Plant Functional Type.
        var_pred_name (list): List of predictor variable names.
        K (int): Number of clusters.
        Nc (int): Number of sites to select from each cluster.
        adict (dict): Dictionary of variables.
        sel_most_PFTs (bool): Whether to select sites with the most PFTs or randomly.
        old_cluster (bool): Whether to use the old clustering approach.
        seed (int): Set random seed. Default value (1000).

    Returns:
        tuple:
            - cluster_dic (dict): Dictionary containing cluster information.
            - distance (float): Sum of squared distances of samples to their closest cluster center.
            - All_selectedID (numpy.ndarray): Array of selected site IDs.
    """
    if "year" in packdata.dims:
        packdata = packdata.mean("year", keep_attrs=True)
    if "Ndep_nhx_pft" in var_pred_name:
        packdata["Ndep_nhx_pft"] = packdata.Ndep_nhx[ipft - 1]
    if "Ndep_noy_pft" in var_pred_name:
        packdata["Ndep_noy_pft"] = packdata.Ndep_noy[ipft - 1]
    if "Pdep_pft" in var_pred_name:
        packdata["Pdep_pft"] = packdata.Pdep[ipft - 1]
    pp = PFT_mask[ipft - 1]
    laix = packdata.LAI0[ipft - 1]
    print("laix.shape:", laix.shape)
    print("pp.shape:", pp.shape)
    pp[laix < 0.01] = np.nan
    var_pred = packdata[var_pred_name] * pp
    df = var_pred.to_dataframe().dropna()
    np.random.seed(seed)
    mod = KMeans(n_clusters=K, random_state=seed)
    CC = mod.fit_predict(df)
    distance = mod.inertia_
    Cluster_output = Series(CC, index=df.index)

    cluster_dic = {}
    All_selectedID = np.empty((0, 2))
    for clus in range(K):
        A = Cluster_output[Cluster_output == clus]
        locations = np.array(A.index.to_list())
        cluster_dic["clus_%.2i_loc" % (clus + 1)] = locations
        # 1.3 Randomly select Nc sites from each cluster
        random.seed(seed)
        # Use the old clustering approach instead.
        if old_cluster:
            if len(locations) > Nc:
                n = random.sample(range(len(locations)), Nc)
                SelectedID = locations[n]
            else:
                SelectedID = locations
        else:
            # otherwise use new approach
            n = max(Nc, int(len(locations) * 0.2))
            random.shuffle(locations)
            if sel_most_PFTs:
                ids = list(map(tuple, locations))
                n_pft = (
                    packdata.PFT_counts.to_series()
                    .loc[ids]
                    .sort_values(ascending=False)
                )
                SelectedID = np.array(list(n_pft.index[:n]))
            else:
                SelectedID = locations[:n]

        print(
            f"Selected {len(SelectedID)} ({len(SelectedID) / len(locations):.2%}) sites in cluster {clus}"
        )
        cluster_dic["clus_%.2i_loc_select" % (clus + 1)] = SelectedID
        All_selectedID = np.append(All_selectedID, SelectedID, axis=0)

    adict["PFT" + str(ipft) + "ClusD"] = cluster_dic
    adict["PFT" + str(ipft) + "trainingID"] = All_selectedID
    return cluster_dic, distance, All_selectedID


def cluster_test(packdata, varlist, logfile):
    """
    Test clustering with different K values for all specified PFTs.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        varlist (dict): Dictionary of variable information.
        logfile (file): File object for logging.

    Returns:
        numpy.ndarray: Array of distances for different K values and PFTs.
    """
    # 1. Clustering analysis
    # Make a mask map according to PFT fractions: nan - <0.00000001; 1 - >=0.00000001
    # I used the output 'VEGET_COV_MAX' by ORCHIDEE-CNP with run the spin-up for 1 year.
    # Please provide the file path and name for PFT fractions with resolution of 2 deg.
    PFT_mask, PFT_mask_lai = genmask.PFT(
        packdata, varlist, varlist["PFTmask"]["cluster_thres"]
    )

    # Predictor metrics
    var_pred_name = varlist["pred"]["clustering"]

    # 2. Use different K value and valuate the clustering results

    adict = {}
    kvalues = list(packdata.Ks)
    kpfts = varlist["clustering"]["pfts"]
    dis_all = np.zeros(shape=(len(kvalues), len(kpfts)))
    for veg in kpfts:
        for kkk in kvalues:
            ClusD, disx, traID = cluster_ana(
                packdata,
                PFT_mask,
                veg,
                var_pred_name,
                kkk,
                10,
                adict,
                sel_most_PFTs=False,
                old_cluster=True,
            )
            dis_all[kvalues.index(kkk), kpfts.index(veg)] = disx

    return dis_all


def cluster_all(
    packdata, varlist, KK, logfile, take_unique, sel_most_PFTs, old_cluster, seed
):
    """
    Perform clustering for all specified PFTs with a chosen K value.

    Args:
        packdata (xarray.Dataset): Dataset containing input variables.
        varlist (dict): Dictionary of variable information.
        KK (int): Chosen K value for clustering.
        logfile (file): File object for logging.

    Returns:
        tuple:
            - IDx (numpy.ndarray): Array of chosen pixel IDs for MLacc.
            - IDloc (numpy.ndarray): Array of cluster locations (for plotting).
            - IDsel (numpy.ndarray): Array of selected cluster locations (for plotting).
    """
    adict = locals()
    kpfts = varlist["clustering"]["pfts"]
    Ncc = varlist["clustering"]["Ncc"]
    PFT_mask, PFT_mask_lai = genmask.PFT(
        packdata, varlist, varlist["PFTmask"]["cluster_thres"]
    )

    packdata["PFT_counts"] = (
        ("lat", "lon"),
        len(PFT_mask_lai) - np.isnan(PFT_mask_lai).sum(axis=0),
    )

    # var_pred_name = varlist["pred"]["clustering"]
    var_pred_name = [k for k, v in packdata.items() if "veget" not in v.dims]

    with ThreadPoolExecutor() as pool:
        pool.map(
            lambda veg: cluster_ana(
                packdata,
                PFT_mask,
                veg,
                var_pred_name,
                KK,
                Ncc[kpfts.index(veg)],
                adict,
                sel_most_PFTs=sel_most_PFTs,
                old_cluster=old_cluster,
            ),
            kpfts,
        )

    # 5. Output the ID
    IDx = np.concatenate([adict["PFT%itrainingID" % ii] for ii in kpfts])

    # 6. If take_unique is False, we are equivalent to the original implementation
    if take_unique:
        IDx = np.unique(IDx, axis=0)
    IDloc = np.array([adict["PFT%iClusD" % ii]["clus_01_loc"] for ii in kpfts])
    IDsel = np.array([adict["PFT%iClusD" % ii]["clus_01_loc_select"] for ii in kpfts])
    return IDx, IDloc, IDsel
