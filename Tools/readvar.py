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


def _get_climate_names(climvar):
    """
    Return climate source variable names and their internal names.

    The legacy configuration uses only ``variables``. New configurations may also
    provide ``rename`` to map source names to SPINacc-internal names.
    """

    source_names = climvar["variables"]
    output_names = climvar.get("rename", source_names)
    if len(source_names) != len(output_names):
        raise ValueError(
            "climate.rename must have the same length as climate.variables"
        )
    return source_names, output_names


def _read_monthly_climate(climvar, source_name, year, nlat, nlon):
    """
    Read one climate variable for one year and aggregate it to monthly means.
    """

    f = Dataset(climvar["sourcepath"] + climvar["filename"] + str(year) + ".nc", "r")
    da = f[source_name][:]
    if isinstance(da, np.ma.masked_array):
        da = da.filled(np.nan)

    if "land" in f[source_name].dimensions:
        land = f["land"][:] - 1
        ntime = len(da)
        uncomp = np.full((ntime, nlat * nlon), np.nan)
        uncomp[:, land] = da
        da = uncomp.reshape((ntime, nlat, nlon))

    days = np.array(calendar.mdays[1:])
    total_days = int(np.nansum(days))
    steps_per_day = max(1, len(da) // total_days) if total_days > 0 else 1
    zstart = 1
    var_month = np.full((12, nlat, nlon), np.nan)
    for month in range(1, 13):
        count = np.nansum(days[:month])
        month_values = da[steps_per_day * (zstart - 1) : steps_per_day * count]
        month_mean = np.mean(month_values, axis=0)
        if isinstance(month_mean, np.ma.masked_array):
            month_mean = month_mean.filled(np.nan)
        var_month[month - 1] = month_mean
        zstart = count + 1

    f.close()
    return var_month


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
    source_clim_names, varname_clim = _get_climate_names(climvar)
    nyear = climvar["year_end"] - climvar["year_start"] + 1
    for source_name, var_name in zip(source_clim_names, varname_clim):
        var_month_year = np.full((nyear, 12, nlat, nlon), np.nan)
        for year in range(climvar["year_start"], climvar["year_end"] + 1):
            check.display("reading %s from year %i" % (source_name, year), logfile)
            var_month_year[year - climvar["year_start"]] = _read_monthly_climate(
                climvar, source_name, year, nlat, nlon
            )
        adict[f"MY{var_name}"] = var_month_year[:]

    packdata.update(
        (k, (["year", "month", "lat", "lon"], adict[f"MY{k}"])) for k in varname_clim
    )

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

    kelvin_variables = set(climvar.get("kelvin_variables", ["Tair", "Tmax", "Tmin"]))
    for var in kelvin_variables.intersection(ds.data_vars):
        ds[var] = ds[var] - 273.15

    if "Tair" not in ds.data_vars and {"Tmax", "Tmin"}.issubset(ds.data_vars):
        ds["Tair"] = (ds["Tmax"] + ds["Tmin"]) / 2

    if "Wind" not in ds.data_vars and {"Wind_E", "Wind_N"}.issubset(ds.data_vars):
        ds["Wind"] = np.hypot(ds["Wind_E"], ds["Wind_N"])

    if "precip" not in ds.data_vars and "Rainf" in ds.data_vars:
        ds["precip"] = ds["Rainf"]
        if "Snowf" in ds.data_vars:
            ds["precip"] = ds["precip"] + ds["Snowf"]

    temp_gs_var = climvar.get("growing_season_temperature", "Tair")
    precip_gs_var = climvar.get("growing_season_precipitation", "precip")
    if temp_gs_var in ds.data_vars and precip_gs_var in ds.data_vars:
        pre = 30 * 24 * 3600 * ds[precip_gs_var].mean("year")
        temp = ds[temp_gs_var].mean("year")
        gs_temp_threshold = climvar.get("gs_temp_threshold", -4)
        Pre_GS_v = np.full((12, nlat, nlon), np.nan)
        Temp_GS_v = np.full((12, nlat, nlon), np.nan)
        GS_length_v = np.full((12, nlat, nlon), np.nan)
        land = np.where(np.isfinite(temp.isel(month=0).values), 1, 0)
        for month in range(1, 13):
            GS_mask = np.zeros(shape=(nlat, nlon))
            maskx = temp.isel(month=month - 1).values
            GS_mask[maskx > gs_temp_threshold] = 1
            Pre_GS_v[month - 1] = GS_mask * pre.isel(month=month - 1).values
            Temp_GS_v[month - 1] = GS_mask * maskx
            GS_length_v[month - 1] = GS_mask * land
        ds["GS_length"] = (("lat", "lon"), np.sum(GS_length_v, axis=0))
        ds["Pre_GS"] = (("lat", "lon"), np.sum(Pre_GS_v, axis=0))
        ds["Temp_GS"] = (("lat", "lon"), np.sum(Temp_GS_v, axis=0))

    flux_variables = set(climvar.get("flux_variables", ["Rainf", "Snowf", "precip"]))
    for var in list(ds.data_vars):
        arr = ds[var]
        if "year" in arr.dims:
            if config.take_year_average:
                arr = arr.mean("year")
            if var in flux_variables:
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
                    # This was previously dropped.
                    Tamp=arr.max("month") - arr.min("month"),
                )
            else:
                stats = {
                    f"{var}_mean": arr.mean("month"),
                    f"{var}_std": arr.std("month"),
                }
            ds = ds.drop_vars(var).assign(stats)

    # 0.3 Interactions between variables
    precip_mean_name = (
        "precip_mean" if "precip_mean" in ds.data_vars else "Rainf_mean"
    )
    if {"Tmean", precip_mean_name}.issubset(ds.data_vars):
        ds["interx1"] = ds.Tmean * ds[precip_mean_name]
    if {"Temp_GS", "Pre_GS"}.issubset(ds.data_vars):
        ds["interx2"] = ds.Temp_GS * ds.Pre_GS

    ds.attrs.update(
        nlat=nlat, nlon=nlon, lat_reso=varlist["lat_reso"], lon_reso=varlist["lon_reso"]
    )

    # range of Ks to be tested, and the final K
    maxK = config.max_kmeans_clusters
    ds.attrs["Ks"] = list(range(2, maxK + 1))
    ds.attrs["K"] = config.kmeans_clusters

    requested_predictors = set(predvar.get("allname", []))
    requested_predictors.update(predvar.get("allname_pft", []))
    requested_predictors.update(
        value for value in predvar.get("clustering", []) if not value.endswith("_pft")
    )
    missing_predictors = sorted(
        value for value in requested_predictors if value not in ds.data_vars
    )
    if missing_predictors:
        available_predictors = sorted(
            value
            for value in ds.data_vars
            if value.endswith("_mean")
            or value.endswith("_std")
            or value in {"Tamp", "Tmax", "Tmean", "Tmin", "Tstd", "Pre_GS", "Temp_GS", "GS_length", "interx1", "interx2"}
        )
        raise KeyError(
            "Configured predictors missing from packdata: "
            + ", ".join(missing_predictors)
            + ". Available climate-derived predictors: "
            + ", ".join(available_predictors)
        )

    return ds
