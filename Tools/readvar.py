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
import xarray


class PackData(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def readvar(varlist, config, logfile):
    """
    Read and process variables from input files.

    Args:
        varlist (dict): Dictionary of variable information.
        config (object): Configuration object.
        logfile (file): File object for logging.

    Returns:
        xarray.Dataset: Dataset containing processed variables.
    """
    adict = locals()
    # 0 initialize latitude and longitudes
    f = Dataset(varlist["coord_ref"], "r")
    nlat = len(f.dimensions["y"])
    nlon = len(f.dimensions["x"])
    packdata = PackData()
    packdata.lat = f["nav_lat"][:, 0]
    packdata.lon = f["nav_lon"][0, :]

    # 0.1 read climate variables
    climvar = varlist["climate"]
    varname_clim = climvar["variables"]
    days = np.array(calendar.mdays[1:])
    nyear = climvar["year_end"] - climvar["year_start"] + 1
    for index in range(len(varname_clim)):
        var_month_year = np.full((nyear, 12, nlat, nlon), np.nan)
        for year in range(climvar["year_start"], climvar["year_end"] + 1):
            check.display(
                "reading %s from year %i" % (varname_clim[index], year), logfile
            )
            # DSG bugfix_start: remove hardcode
            # filename='crujra_twodeg_v2_'+str(year)+'.nc'
            f = Dataset(
                climvar["sourcepath"] + climvar["filename"] + str(year) + ".nc", "r"
            )
            # DSG bugfix_end
            da = f[varname_clim[index]][:]
            # DSG: fix to read in compressed netCDF files
            if "land" in f[varname_clim[index]].dimensions:
                land = f["land"][:] - 1
                ntime = len(da)
                uncomp = np.arr.masked_all((ntime, nlat * nlon))
                uncomp[:, land] = da
                da = uncomp.reshape((ntime, nlat, nlon))
            # DSG: end

            # calculate the monthly value from 6h data
            zstart = 1
            var_month = np.full((12, nlat, nlon), np.nan)
            count = 0
            for month in range(1, 13):
                count = np.nansum(days[:month])
                mkk = np.mean(da[4 * (zstart - 1) : 4 * count], axis=0)
                # mkk[da[0]==mask]=np.nan
                var_month[month - 1] = mkk.filled(np.nan)
                zstart = count + 1

            var_month_year[year - climvar["year_start"]] = var_month
        adict["MY%s" % varname_clim[index]] = var_month_year[:]

    # 0.1.1 Tair (Tmax, Tmin, Tmean,Tstd,AMT)
    packdata.Tair = (["year", "month", "lat", "lon"], adict["MYTair"] - 273.15)

    # 0.1.2 Other climatic variables (Rainf,Snowf,Qair,Psurf,SWdown,LWdown)
    packdata.update(
        (k, (["year", "month", "lat", "lon"], adict[f"MY{k}"])) for k in varname_clim
    )

    # 0.1.3 P and T for growing season (Pre_GS, Temp_GS, GS_length)
    pre = 30 * 24 * 3600 * np.mean(adict["MYRainf"], axis=0)
    temp = np.mean(packdata.Tair[1], axis=0)
    Pre_GS_v = np.full((12, nlat, nlon), np.nan)
    Temp_GS_v = np.full((12, nlat, nlon), np.nan)
    GS_length_v = np.full((12, nlat, nlon), np.nan)
    land = adict["MYTair"][0][0]
    land[land > 1] = 1
    for month in range(1, 13):
        GS_mask = np.zeros(shape=(nlat, nlon))
        maskx = temp[month - 1]
        # temperature > 4 degree is growing season
        GS_mask[maskx > -4] = 1
        Pre_GS_v[month - 1] = GS_mask * pre[month - 1]
        Temp_GS_v[month - 1] = GS_mask * temp[month - 1]
        GS_length_v[month - 1] = GS_mask * land
    packdata.GS_length = (("lat", "lon"), np.sum(GS_length_v, axis=0))
    packdata.Pre_GS = (("lat", "lon"), np.sum(Pre_GS_v, axis=0))
    packdata.Temp_GS = (("lat", "lon"), np.sum(Temp_GS_v, axis=0))

    # 0.2 read other variables, including Edaphic variables, N and P deposition variables
    predvar = varlist["pred"]
    for ipred in predvar.keys():
        if ipred[:3] == "var":
            f = Dataset(predvar[ipred]["sourcefile"], "r")
            vname = predvar[ipred]["variables"]
            if "rename" in predvar[ipred].keys():
                rename = predvar[ipred]["rename"]
            else:
                rename = vname
            for ivar in range(len(vname)):
                check.display("reading %s..." % vname[ivar], logfile)
                if (
                    vname[ivar] == "LAI"
                    or vname[ivar] == "NPP"
                    or vname[ivar] == "P_DEPOSITION"
                ) and len(f[vname[ivar]].shape) > 3:
                    # one can modify here to use annual mean
                    check.verbose(
                        "warning: using %s at the last year" % vname[ivar], logfile
                    )
                    da = np.squeeze(f[vname[ivar]][-1])
                # N deposition fluxes miss PFT dimension
                elif (
                    vname[ivar] == "NOY_DEPOSITION" or vname[ivar] == "NHX_DEPOSITION"
                ) and len(f[vname[ivar]].shape) > 2:
                    # one can modify here to use annual mean
                    check.verbose(
                        "warning: using %s at the last year" % vname[ivar], logfile
                    )
                    da = np.squeeze(f[vname[ivar]][-1])
                elif (vname[ivar] == "clayfraction") and len(f[vname[ivar]].shape) > 2:
                    # if clayfraction is discretized vertically, use 1st soil layer
                    check.verbose(
                        "warning: using only the first soil layer value for %s"
                        % vname[ivar],
                        logfile,
                    )
                    da = np.squeeze(f[vname[ivar]][0])
                else:
                    da = np.squeeze(f[vname[ivar]][:])
                if "missing_value" in predvar[ipred].keys():
                    da[da == predvar[ipred]["missing_value"]] = np.nan
                if isinstance(da, np.ma.masked_array):
                    da = da.filled(np.nan)
                packdata[rename[ivar]] = (["veget", "lat", "lon"][-da.ndim :], da)

    ds = xarray.Dataset(packdata)

    for var, arr in ds.data_vars.items():
        if "year" in arr.dims:
            # arr = arr.mean("year")
            if var in ("Rainf", "Snowf"):
                stats = {
                    f"{var}_mean": 365 * 24 * 3600 * arr.mean("month"),
                    f"{var}_std": 30 * 24 * 3600 * arr.std("month"),
                }
            elif var == "Tair":
                stats = dict(
                    Tmean=arr.mean("month"),
                    Tstd=arr.std("month"),
                    Tmin=arr.min("month"),
                    Tmax=arr.max("month"),
                    Tamp=arr.max("month") - arr.min("month"),
                )
            else:
                stats = {
                    f"{var}_mean": arr.mean("month"),
                    f"{var}_std": arr.std("month"),
                }
            ds = ds.drop_vars(var).assign(stats)

    # 0.3 Interactions between variables
    ds["interx1"] = ds.Tmean * ds.Rainf_mean
    ds["interx2"] = ds.Temp_GS * ds.Pre_GS

    ds.attrs.update(
        nlat=nlat, nlon=nlon, lat_reso=varlist["lat_reso"], lon_reso=varlist["lon_reso"]
    )

    # range of Ks to be tested, and the final K
    maxK = config.max_kmeans_clusters
    ds.attrs["Ks"] = list(range(2, maxK + 1))
    ds.attrs["K"] = config.kmeans_clusters

    return ds
