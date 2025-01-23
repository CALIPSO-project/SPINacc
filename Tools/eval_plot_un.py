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

import re


def get_suffix_mapping(varname):
    """
    Get the suffix mapping for the pool name.

    Args:
        varname (str): Variable name (e.g., 'biomass_01', 'biomass_01_n', 'biomass_01_p').

    Returns:
        str: Pool name (e.g., 'carbon', 'nitrogen', 'phosphorus').

    """
    # Regex to capture the suffix (if present)
    match = re.search(r"_(n|p)$", varname)
    if match:
        suffix = match.group(1)
        return {"n": "nitrogen", "p": "phosphorus"}.get(
            suffix, "carbon"
        )  # Default to 'carbon' if suffix is unrecognized
    return "carbon"  # Default to 'carbon' if no suffix is found


def plot_metric(data_path, npfts, ipool, subLabel, xTickLabel):
    """
    Plot the evaluation metrics for the machine learning predictions.

    Args:
        data_path (Path): Path to the data directory.
        npfts (int): Number of Plant Functional Types (14).
        ipool (str): Pool name (e.g., 'som', 'biomass', 'litter', 'microbe').
        subLabel (list): List of sub-level item names (e.g., ['Cpool'] or ['Cpool', 'Npool', 'Ppool']).
        xTickLabel (list): List of subpool names (e.g., 'Active', 'Passive', 'Slow').

    Returns:
        None

    Trunk example:
    plot_metric(data_path, 14, 'som', ['Cpool'], 1, 3, ['Active', 'Passive', 'Slow'])

    CNP2 example:
    plot_metric(data_path, 14, 'som', ['Cpool', 'Npool', 'Ppool'], 1, 3, ['Active', 'ChemProtect', 'PhysProtect'])

    """
    subps = len(xTickLabel)
    # Get length of ['Cpool'] (1) or ['Cpool', 'Npool', 'Ppool'] (3)
    loop_n = len(subLabel)
    # Read the csv output file as DataFrame and set multi-index containing ipft, ivar, and subpool
    df = pd.read_csv(data_path / "MLacc_results.csv", index_col=[0, 1, 2])
    df = df.loc[ipool].round(2)

    # Pre-processing for pools that have C, N and P components
    # Create new 'discriminator' column containing component and stack C, N, P components.
    if loop_n == 3:
        if ipool == "lignin":
            loop_n = 1
        elif ipool == "som":
            df["discrim"] = df["varname"].str.split("_").str[0]
            df.set_index(["discrim"], append=True, inplace=True)
        elif ipool == "biomass" or ipool == "microbe" or ipool == "litter":
            df["discrim"] = df["varname"].apply(get_suffix_mapping)
            df.set_index(["discrim"], append=True, inplace=True)

    # (ipft, ivar) stacked by carbon, nitrogen, phosphorus
    # The sort_index() function only applies when the index is a MultiIndex - i.e. when C, N, P are stacked
    # Has no effect otherwise.
    R22 = df["R2"].unstack(0).sort_index(level=1).values
    slope = df["slope"].unstack(0).sort_index(level=1).values
    dNRMSE = df["dNRMSE"].unstack(0).sort_index(level=1).values
    yTickLabel = [f"PFT{pft:02d}" for pft in range(2, npfts + 2)]
    fonts = 7
    # slope=slope[0:npfts,0:subps]
    # dNRMSE=dNRMSE[0:npfts,0:subps]
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

    for n in range(0, loop_n):
        R22_n = R22[n * npfts : (n + 1) * npfts, :]
        slope_n = slope[n * npfts : (n + 1) * npfts, :]
        dNRMSE_n = dNRMSE[n * npfts : (n + 1) * npfts, :]

        fig, axs = plt.subplots(nrows=3, figsize=(8, 18))
        axs[0].imshow(R22_n, vmin=0.5, vmax=1, cmap=mymap_R2)
        for jj in range(0, subps):
            # print(jj)
            for ii in range(0, npfts):
                # print(R22_n[ii,jj])
                axs[0].text(-0.5 + jj, ii, str(R22_n[ii, jj]), size=fonts, color="k")

        my_x_ticks = np.arange(subps)
        axs[0].set_xticks(my_x_ticks)
        # axs[0].set_xticklabels([""])
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
        # slope
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
        # axs[1].set_xticklabels([""])
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
        plt.savefig(data_path / f"Eval_all_{ipool}{subLabel[n]}.png")
        plt.close("all")
    return
