from Tools import *


def write(varlist, resultpath, IDx):

    # select mode to build forcing and restart files
    # possible modes: compressed, regular, aligned
    mode = "compressed"

    # define indices of selected pixels using the first forcing year
    nc = Dataset(
        varlist["climate"]["sourcepath"]
        + varlist["climate"]["filename"]
        + str(varlist["climate"]["year_start"])
        + ".nc"
    )
    if "land" in nc.dimensions:
        land = nc.variables["land"][:]
    lat = nc.variables[
        (
            "lat"
            if "lat" in nc.variables
            else "latitude" if "latitude" in nc.variables else "nav_lat"
        )
    ][
        :
    ]  #
    lon = nc.variables[
        (
            "lon"
            if "lon" in nc.variables
            else "longitude" if "longitude" in nc.variables else "nav_lon"
        )
    ][:]
    if len(lat.shape) > 1:
        lat = lat[:, 0]
    if len(lon.shape) > 1:
        lon = lon[0, :]
    ilats = list()
    ilons = list()
    ilands = list()
    xlands = list()
    for xlat, xlon in IDx:
        ilat = np.where(xlat == lat)[0][0]
        ilon = np.where(xlon == lon)[0][0]
        xland = ilat * len(lon) + ilon + 1
        if xland in xlands:
            continue
        if "land" in nc.dimensions:
            iland = np.where(xland == land)[0][0]
        else:
            iland = None
        ilats.append(ilat)
        ilons.append(ilon)
        ilands.append(iland)
        xlands.append(xland)
    ilats = np.array(ilats)
    ilons = np.array(ilons)
    ilands = np.array(ilands)
    xlands = np.array(xlands)
    nc.close()

    # build compressed forcing
    if mode == "compressed":
        for year in range(
            varlist["climate"]["year_start"], varlist["climate"]["year_end"] + 1
        ):
            print("Building compressed forcing for year %s" % year)
            nc = Dataset(
                varlist["climate"]["sourcepath"]
                + varlist["climate"]["filename"]
                + str(year)
                + ".nc"
            )
            ncout = Dataset(resultpath + "forcing_compressed_" + str(year) + ".nc", "w")
            for dim in nc.dimensions:
                newdim = (
                    "x"
                    if dim in ["lon", "longitude"]
                    else "y" if dim in ["lat", "latitude"] else dim
                )
                newsize = (
                    len(xlands)
                    if dim == "land"
                    else (
                        None
                        if nc.dimensions[dim].isunlimited()
                        else len(nc.dimensions[dim])
                    )
                )
                ncout.createDimension(newdim, newsize)
            if "land" not in ncout.dimensions:
                ncout.createDimension("land", len(xlands))
                ncout.createVariable("land", "i4", ("land",))
                ncout.variables["land"].setncattr("compress", "y x")
                ncout.variables["land"][:] = xlands
            for var in nc.variables:
                print(var)
                if len(nc.variables[var].dimensions) == 3:
                    newdims = (nc.variables[var].dimensions[0], "land")
                elif (
                    "land" not in nc.dimensions
                    and len(nc.variables[var].dimensions) == 2
                ):
                    newdims = ("y", "x")
                else:
                    newdims = nc.variables[var].dimensions
                if var in ["land", "nav_lon", "nav_lat", "time", "timeplussix"]:
                    ncout.createVariable(var, nc.variables[var].dtype, newdims)
                else:
                    ncout.createVariable(
                        var,
                        nc.variables[var].dtype,
                        newdims,
                        fill_value=9.96921e36,
                        zlib=True,
                    )
                atts = dict()
                for key, value in nc.variables[var].__dict__.items():
                    if key not in ["_FillValue", "missing_value"]:
                        atts[key] = value
                ncout.variables[var].setncatts(atts)
                if var == "land":
                    ncout.variables[var][:] = xlands
                elif len(nc.variables[var].dimensions) == 3:
                    ncdata = nc.variables[var][:]
                    ncout.variables[var][:] = ncdata.reshape(
                        len(ncdata), len(lat) * len(lon)
                    )[:, xlands - 1]
                elif nc.variables[var].dimensions[-1] == "land":
                    ncdata = nc.variables[var][:]
                    ncout.variables[var][:] = ncdata[:, ilands]
                elif var.lower() in ["contfrac"]:
                    ncdata = nc.variables[var][:]
                    data = np.ma.masked_all((len(lat), len(lon)))
                    data[ilats, ilons] = ncdata[ilats, ilons]
                    ncout.variables[var][:] = data
                else:
                    ncout.variables[var][:] = nc.variables[var][:]
            ncout.close()
            nc.close()

    # build regular forcing
    if mode == "regular":
        for year in range(
            varlist["climate"]["year_start"], varlist["climate"]["year_end"] + 1
        ):
            print("Building regular forcing for year %s" % year)
            nc = Dataset(
                varlist["climate"]["sourcepath"]
                + varlist["climate"]["filename"]
                + str(year)
                + ".nc"
            )
            ncout = Dataset(resultpath + "forcing_regular_" + str(year) + ".nc", "w")
            for dim in nc.dimensions:
                if dim == "land":
                    continue
                newdim = (
                    "lon"
                    if dim in ["lon", "longitude", "x"]
                    else "lat" if dim in ["lat", "latitude", "y"] else dim
                )
                newsize = (
                    len(lat)
                    if newdim == "lat"
                    else (
                        len(lon)
                        if newdim == "lon"
                        else (
                            None
                            if nc.dimensions[dim].isunlimited()
                            else len(nc.dimensions[dim])
                        )
                    )
                )
                ncout.createDimension(newdim, newsize)
            for var in nc.variables:
                print(var)
                if var == "land":
                    continue
                elif (
                    len(nc.variables[var].dimensions) == 3
                    or "land" in nc.variables[var].dimensions
                ):
                    newdims = (nc.variables[var].dimensions[0], "lat", "lon")
                elif len(nc.variables[var].dimensions) == 2:
                    newdims = ("lat", "lon")
                elif nc.variables[var].dimensions[0] in ["lon", "longitude", "x"]:
                    newdims = ("lon",)
                elif nc.variables[var].dimensions[0] in ["lat", "latitude", "y"]:
                    newdims = ("lat",)
                else:
                    newdims = nc.variables[var].dimensions
                if len(newdims) > 2:
                    ncout.createVariable(
                        var,
                        nc.variables[var].dtype,
                        newdims,
                        fill_value=9.96921e36,
                        zlib=True,
                    )
                else:
                    ncout.createVariable(
                        var, nc.variables[var].dtype, newdims, fill_value=9.96921e36
                    )
                atts = dict()
                for key, value in nc.variables[var].__dict__.items():
                    if key not in ["_FillValue", "missing_value"]:
                        atts[key] = value
                ncout.variables[var].setncatts(atts)
                if nc.variables[var].dimensions[-1] == "land":
                    ncdata = nc.variables[var][:]
                    data = np.ma.masked_all((len(ncdata), len(lat) * len(lon)))
                    data[:, xlands - 1] = ncdata[:, ilands]
                    ncout.variables[var][:] = data.reshape(
                        (len(nc.variables[var]), len(lat), len(lon))
                    )
                elif len(nc.variables[var].dimensions) == 3:
                    ncdata = nc.variables[var][:]
                    data = np.ma.masked_all((len(ncdata), len(lat), len(lon)))
                    data[:, ilats, ilons] = ncdata[:, ilats, ilons]
                    ncout.variables[var][:] = data
                elif var.lower() in ["contfrac"]:
                    ncdata = nc.variables[var][:]
                    data = np.ma.masked_all((len(lat), len(lon)))
                    data[ilats, ilons] = ncdata[ilats, ilons]
                    ncout.variables[var][:] = data
                else:
                    ncout.variables[var][:] = nc.variables[var][:]
            ncout.close()
            nc.close()

    # build regular restart files (also used with compressed forcing)
    if mode == "compressed" or mode == "regular":
        for path in varlist["restart"]:
            print("Building regular restart file for", path)
            nc = Dataset(path)
            ncout = Dataset(resultpath + os.path.split(path)[-1], "w")
            for dim in nc.dimensions:
                ncout.createDimension(
                    dim,
                    (
                        None
                        if nc.dimensions[dim].isunlimited()
                        else len(nc.dimensions[dim])
                    ),
                )
            for var in nc.variables:
                ncout.createVariable(
                    var, nc.variables[var].dtype, nc.variables[var].dimensions
                )
                ncout.variables[var].setncatts(nc.variables[var].__dict__)
                if len(nc.variables[var].dimensions) > 2:
                    ncdata = nc.variables[var][:]
                    data = np.ma.masked_all(ncdata.shape)
                    data[..., ilats, ilons] = ncdata[..., ilats, ilons]
                    ncout.variables[var][:] = data
                else:
                    ncout.variables[var][:] = nc.variables[var][:]
            nc.close()
            ncout.close()

    # build aligned forcing
    if mode == "aligned":
        nlat = int(np.ceil((len(IDx) / 2) ** 0.5))
        nlon = nlat * 2
        step = 180.0 / nlat
        plat = np.arange(90 - step / 2.0, -90, -step)
        plon = np.arange(-180 + step / 2.0, 180, step)
        pnavlon, pnavlat = np.meshgrid(plon, plat)
        alats = list()
        alons = list()
        for idx in range(len(xlands)):
            alats.append(idx // nlon)
            alons.append(idx % nlon)
        alats = np.array(alats)
        alons = np.array(alons)
        for year in range(
            varlist["climate"]["year_start"], varlist["climate"]["year_end"] + 1
        ):
            print("Building aligned forcing for year %s" % year)
            nc = Dataset(
                varlist["climate"]["sourcepath"]
                + varlist["climate"]["filename"]
                + str(year)
                + ".nc"
            )
            ncout = Dataset(resultpath + "forcing_aligned_" + str(year) + ".nc", "w")
            for dim in nc.dimensions:
                if dim == "land":
                    continue
                newdim = (
                    "lon"
                    if dim in ["lon", "longitude", "x"]
                    else "lat" if dim in ["lat", "latitude", "y"] else dim
                )
                newsize = (
                    len(plat)
                    if newdim == "lat"
                    else (
                        len(plon)
                        if newdim == "lon"
                        else (
                            None
                            if nc.dimensions[dim].isunlimited()
                            else len(nc.dimensions[dim])
                        )
                    )
                )
                ncout.createDimension(newdim, newsize)
            for var in nc.variables:
                print(var)
                if var == "land":
                    continue
                elif (
                    len(nc.variables[var].dimensions) == 3
                    or "land" in nc.variables[var].dimensions
                ):
                    newdims = (nc.variables[var].dimensions[0], "lat", "lon")
                elif len(nc.variables[var].dimensions) == 2:
                    newdims = ("lat", "lon")
                elif nc.variables[var].dimensions[0] in ["lon", "longitude", "x"]:
                    newdims = ("lon",)
                elif nc.variables[var].dimensions[0] in ["lat", "latitude", "y"]:
                    newdims = ("lat",)
                else:
                    newdims = nc.variables[var].dimensions
                if len(newdims) > 2:
                    ncout.createVariable(
                        var,
                        nc.variables[var].dtype,
                        newdims,
                        fill_value=9.96921e36,
                        zlib=True,
                    )
                else:
                    ncout.createVariable(
                        var, nc.variables[var].dtype, newdims, fill_value=9.96921e36
                    )
                atts = dict()
                for key, value in nc.variables[var].__dict__.items():
                    if key not in ["_FillValue", "missing_value"]:
                        atts[key] = value
                ncout.variables[var].setncatts(atts)
                if len(nc.variables[var].dimensions) == 3:
                    ncdata = nc.variables[var][:]
                    data = np.ma.masked_all((len(ncdata), nlat, nlon))
                    data[:, alats, alons] = ncdata[:, ilats, ilons]
                    ncout.variables[var][:] = data
                    for idx in range(len(xlands)):
                        ncout.variables[var][:, idx // nlon, idx % nlon] = nc.variables[
                            var
                        ][:, ilats[idx], ilons[idx]]
                elif nc.variables[var].dimensions[-1] == "land":
                    ncdata = nc.variables[var][:]
                    data = np.ma.masked_all((len(ncdata), nlat, nlon))
                    data[:, alats, alons] = ncdata[:, ilands]
                    ncout.variables[var][:] = data
                    for idx in range(len(xlands)):
                        ncout.variables[var][:, idx // nlon, idx % nlon] = nc.variables[
                            var
                        ][:, ilands[idx]]
                elif var == "nav_lat":
                    ncout.variables[var][:] = pnavlat
                elif var == "nav_lon":
                    ncout.variables[var][:] = pnavlon
                elif var == "lat":
                    ncout.variables[var][:] = plat
                elif var == "lon":
                    ncout.variables[var][:] = plon
                elif var == "contfrac":
                    ncdata = nc.variables[var][:]
                    data = np.ma.masked_all((nlat, nlon))
                    data[alats, alons] = ncdata[ilats, ilons]
                    ncout.variables[var][:] = data
                else:
                    ncout.variables[var][:] = nc.variables[var][:]
            ncout.close()
            nc.close()

    # build aligned restart files
    if mode == "aligned":
        for path in varlist["restart"]:
            print("Building aligned restart file for", path)
            nc = Dataset(path)
            ncout = Dataset(resultpath + os.path.split(path)[-1], "w")
            for dim in nc.dimensions:
                newsize = (
                    len(plat)
                    if dim in ["lat", "latitude", "y"]
                    else (
                        len(plon)
                        if dim in ["lon", "longitude", "x"]
                        else (
                            None
                            if nc.dimensions[dim].isunlimited()
                            else len(nc.dimensions[dim])
                        )
                    )
                )
                ncout.createDimension(dim, newsize)
            for var in nc.variables:
                ncout.createVariable(
                    var, nc.variables[var].dtype, nc.variables[var].dimensions
                )
                ncout.variables[var].setncatts(nc.variables[var].__dict__)
                if var in ["nav_lat", "lat", "LAT", "latitude", "y"]:
                    if len(nc.variables[var].dimensions) == 2:
                        ncout.variables[var][:] = pnavlat
                    else:
                        ncout.variables[var][:] = plat
                elif var in ["nav_lon", "lon", "LON", "longitude", "x"]:
                    if len(nc.variables[var].dimensions) == 2:
                        ncout.variables[var][:] = pnavlon
                    else:
                        ncout.variables[var][:] = plon
                elif nc.variables[var].dimensions[-1] in [
                    "lon",
                    "longitude",
                    "x",
                ] and nc.variables[var].dimensions[-2] in ["lat", "latitude", "y"]:
                    for idx in range(len(xlands)):
                        ncout.variables[var][..., idx // nlon, idx % nlon] = (
                            nc.variables[var][..., ilats[idx], ilons[idx]]
                        )
                else:
                    ncout.variables[var][:] = nc.variables[var][:]
            nc.close()
            ncout.close()
