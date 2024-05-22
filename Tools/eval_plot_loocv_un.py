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

##@param[in]   data_path            config[5] resultpath
##@param[in]   npfts                number of PFT
##@param[in]   ipool                som, biomass, litter
##@param[in]   n_cnp                c-only:1 cn:2 cnp:3
##@param[in]   xTickLabel           The names of subpools

# xTickLabel = ['Active','Passive','Slow','Surface']


# data_path='/home/orchidee04/ysun/MLacc_Python_Tool/'
def plot_metric(data_path, npfts, ipool, subLabel, dims, sect_n, xTickLabel):
    print(dims)
    subps = len(xTickLabel)
    # print(subps)
    loop_n = len(subLabel)
    shape = np.array([npfts, subps])
    R22 = np.genfromtxt(data_path + ipool + "_loocv_R2.txt", delimiter=",")[:, :-1]
    slope = np.genfromtxt(data_path + ipool + "_loocv_slope.txt", delimiter=",")[:, :-1]
    dNRMSE = np.genfromtxt(data_path + ipool + "_loocv_dNRMSE.txt", delimiter=",")[
        :, :-1
    ]

    # print(R22)
    yTickLabel = [
        "PFT02",
        "PFT03",
        "PFT04",
        "PFT05",
        "PFT06",
        "PFT07",
        "PFT08",
        "PFT09",
        "PFT10",
        "PFT11",
        "PFT12",
        "PFT13",
        "PFT14",
        "PFT15",
    ]
    yTickLabel = yTickLabel[0:npfts]
    # R22=R22[:,0:subps]
    fonts = 7
    # slope=slope[:,0:subps]
    # dNRMSE=dNRMSE[:,0:subps]
    # titles=['Cpools','Npools','Ppools'];
    colors1 = plt.cm.YlGn(np.linspace(0, 1, 128))
    colors2 = plt.cm.YlGn_r(np.linspace(0, 1, 128))
    colors = np.vstack((colors1, colors2))
    mycolor_R2 = ["maroon", "tomato", "gold", "limegreen", "forestgreen"]
    mycolor_slope = [
        "maroon",
        "tomato",
        "gold",
        "limegreen",
        "forestgreen",
        "forestgreen",
        "limegreen",
        "gold",
        "tomato",
        "maroon",
    ]
    mycolor_rmse = ["forestgreen", "limegreen", "gold", "tomato", "maroon"]
    mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
    mymap_R2 = mcolors.LinearSegmentedColormap.from_list("my_list", mycolor_R2, N=5)
    mymap_slope = mcolors.LinearSegmentedColormap.from_list(
        "my_list", mycolor_slope, N=10
    )
    mymap_rmse = mcolors.LinearSegmentedColormap.from_list("mylist", mycolor_rmse, N=5)
    # clip
    print(ipool)
    print(sect_n)
    if sect_n > 1:
        jq = sect_n * npfts
    else:
        jq = npfts
    for n in range(0, loop_n):
        print(n)
        print(dims)
        if dims[0] == 0:
            R22_n = R22[n * jq : (n + 1) * jq, :]
            slope_n = slope[n * jq : (n + 1) * jq, :]
            dNRMSE_n = dNRMSE[n * jq : (n + 1) * jq, :]
        else:
            R22_n = R22[n * subps : (n + 1) * subps, :]
            slope_n = slope[n * subps : (n + 1) * subps, :]
            dNRMSE_n = dNRMSE[n * subps : (n + 1) * subps, :]
        print(R22_n)
        R22_n = np.transpose(R22_n, dims).reshape(shape, order="F")
        print(R22_n)
        slope_n = np.transpose(slope_n, dims).reshape(shape, order="F")
        dNRMSE_n = np.transpose(dNRMSE_n, dims).reshape(shape, order="F")
        fig, axs = plt.subplots(nrows=3, figsize=(8, 18))
        axs[0].imshow(R22_n, vmin=0.5, vmax=1, cmap=mymap_R2)
        for jj in range(0, subps):
            for ii in range(0, npfts):
                axs[0].text(-0.5 + jj, ii, str(R22_n[ii, jj]), size=fonts, color="k")

        my_x_ticks = np.arange(subps)
        axs[0].set_xticks(my_x_ticks)
        axs[0].set_xticklabels([""])
        my_y_ticks = np.arange(npfts)
        axs[0].set_yticks(my_y_ticks)
        axs[0].set_yticklabels(yTickLabel)
        axs[0].set_title("R2_" + subLabel[n])
        fig.subplots_adjust(right=0.9)
        l = 0.92
        b = 0.66
        w = 0.015
        h = 0.22
        rect = [l, b, w, h]
        cbar_ax = fig.add_axes(rect)
        sc = axs[0].imshow(R22_n, vmin=0.5, vmax=1, cmap=mymap_R2)
        plt.colorbar(sc, cax=cbar_ax)
        # sloop
        axs[1].imshow(slope_n, vmin=0.75, vmax=1.25, cmap=mymap_slope)
        for jj in range(0, subps):
            for ii in range(0, npfts):
                axs[1].text(
                    -0.5 + jj,
                    ii,
                    str(slope_n[ii, jj]),
                    size=fonts,
                    color="k",
                    weight="bold",
                )
        my_x_ticks = np.arange(subps)
        axs[1].set_xticks(my_x_ticks)
        axs[1].set_xticklabels([""])
        my_y_ticks = np.arange(npfts)
        axs[1].set_yticks(my_y_ticks)
        axs[1].set_yticklabels(yTickLabel)
        axs[1].set_title("slope_" + subLabel[n])
        fig.subplots_adjust(right=0.9)
        l = 0.92
        b = 0.39
        w = 0.015
        h = 0.22
        rect = [l, b, w, h]
        cbar_ax = fig.add_axes(rect)
        sc = axs[1].imshow(slope_n, vmin=0.75, vmax=1.25, cmap=mymap_slope)
        plt.colorbar(sc, cax=cbar_ax)

        # remse
        axs[2].imshow(dNRMSE_n, vmin=0, vmax=0.25, cmap=mymap_rmse)
        for jj in range(0, subps):
            for ii in range(0, npfts):
                axs[2].text(
                    -0.5 + jj,
                    ii,
                    str(dNRMSE_n[ii, jj]),
                    size=fonts,
                    color="k",
                    weight="bold",
                )
        my_x_ticks = np.arange(subps)
        axs[2].set_xticks(my_x_ticks)
        axs[2].set_xticklabels(xTickLabel, rotation=60)
        my_y_ticks = np.arange(npfts)
        axs[2].set_yticks(my_y_ticks)
        axs[2].set_yticklabels(yTickLabel)
        axs[2].set_title("dNRMSE_" + subLabel[n])
        fig.subplots_adjust(right=0.9)
        l = 0.92
        b = 0.12
        w = 0.015
        h = 0.22
        rect = [l, b, w, h]
        cbar_ax = fig.add_axes(rect)
        sc = axs[2].imshow(dNRMSE_n, vmin=0, vmax=0.25, cmap=mymap_rmse)
        plt.colorbar(sc, cax=cbar_ax)

        # else:
        #     fig,axs = plt.subplots(nrows = 3, ncols = n_cnp,figsize = (8,18))
        #     for kn in range(0,n_cnp):
        #         Rm=R22[kn*npfts:(kn+1)*npfts,:]
        #         axs[0,kn].imshow(Rm,vmin = 0.5,vmax = 1,cmap = mymap_rmse)
        #         for jj in range(0,subps):
        #             for ii in range(0,npfts):
        #                 axs[0,kn].text(-0.5+jj, ii,str(Rm[ii,jj]), size = fonts, color='k')

        #         my_x_ticks = np.arange(subps)
        #         axs[0,kn].set_xticks(my_x_ticks)
        #         axs[0,kn].set_xticklabels( xTickLabel,rotation=60 )
        #         my_y_ticks = np.arange(npfts)
        #         axs[0,kn].set_yticks(my_y_ticks)
        #         axs[0,kn].set_yticklabels( yTickLabel)
        #         axs[0,kn].set_title('R2_'+titles[kn])
        #     fig.subplots_adjust(right=0.9)
        #     l = 0.92
        #     b = 0.66
        #     w = 0.015
        #     h = 0.22
        #     rect = [l,b,w,h]
        #     cbar_ax = fig.add_axes(rect)
        #     sc = axs[0,n_cnp-1].imshow(Rm,vmin = 0.5,vmax = 1,cmap = mymap_rmse)
        #     plt.colorbar(sc,cax=cbar_ax)

        # slope
        #    for kn in range(0,n_cnp):
        #         sl=slope[kn*npfts:(kn+1)*npfts,:]
        #         axs[1,kn].imshow(sl,vmin = 0.75,vmax = 1.25,cmap = mymap_slope)
        #         for jj in range(0,subps):
        #             for ii in range(0,npfts):
        #                 axs[1,kn].text(-0.5+jj, ii,str(sl[ii,jj]), size = fonts, color='k')

        #         my_x_ticks = np.arange(subps)
        #         axs[1,kn].set_xticks(my_x_ticks)
        #         axs[1,kn].set_xticklabels( xTickLabel,rotation=60 )
        #         my_y_ticks = np.arange(npfts)
        #         axs[1,kn].set_yticks(my_y_ticks)
        #         axs[1,kn].set_yticklabels( yTickLabel)
        #         axs[1,kn].set_title('slope_'+titles[kn])
        #     fig.subplots_adjust(right=0.9)
        #     l = 0.92
        #     b = 0.39
        #     w = 0.015
        #     h = 0.22
        #     rect = [l,b,w,h]
        #     cbar_ax = fig.add_axes(rect)
        #     sc = axs[0,n_cnp-1].imshow(sl,vmin = 0.75,vmax = 1.25,cmap = mymap_slope)
        #     plt.colorbar(sc,cax=cbar_ax)

        # reMSE
        #     for kn in range(0,n_cnp):
        #         remse=dNRMSE[kn*npfts:(kn+1)*npfts,:]
        #         axs[2,kn].imshow(remse,vmin = 0,vmax = 0.25,cmap = mymap_rmse)
        #         for jj in range(0,subps):
        #             for ii in range(0,npfts):
        #                 axs[2,kn].text(-0.5+jj, ii,str(remse[ii,jj]), size = fonts, color='k')

        #         my_x_ticks = np.arange(subps)
        #         axs[2,kn].set_xticks(my_x_ticks)
        #         axs[2,kn].set_xticklabels( xTickLabel,rotation=60 )
        #         my_y_ticks = np.arange(npfts)
        #         axs[2,kn].set_yticks(my_y_ticks)
        #         axs[2,kn].set_yticklabels( yTickLabel)
        #         axs[2,kn].set_title('dNRMSE_'+titles[kn])
        #     fig.subplots_adjust(right=0.9)
        #     l = 0.92
        #     b = 0.12
        #     w = 0.015
        #     h = 0.22
        #     rect = [l,b,w,h]
        #     cbar_ax = fig.add_axes(rect)
        #     sc = axs[2,n_cnp-1].imshow(remse,vmin = 0,vmax = 0.25,cmap = mymap_rmse)
        #     plt.colorbar(sc,cax=cbar_ax)

        plt.savefig(data_path + "Eval_all_loocv_" + ipool + subLabel[n] + ".png")
        plt.close("all")
    return
