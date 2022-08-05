from Tools import *

def write(varlist,resultpath,IDx):

  # define pseudo-grid
  nlat = int(np.ceil((len(IDx)/2)**0.5))
  nlon = nlat*2
  step = 180./nlat
  plat = np.arange(90 - step/2., -90, -step)
  plon = np.arange(-180 + step/2., 180, step)
  pnavlon, pnavlat = np.meshgrid(plon, plat)

  # define indices of selected pixels using the first forcing year
  nc = Dataset(varlist["climate"]["sourcepath"] + varlist["climate"]["filename"] + str(varlist["climate"]["year_start"]) + ".nc")
  if "land" in nc.dimensions: land = nc.variables["land"][:]
  lat = nc.variables["lat" if "lat" in nc.variables else "latitude" if "latitude" in nc.variables else "nav_lat"][:]
  lon = nc.variables["lon" if "lon" in nc.variables else "longitude" if "longitude" in nc.variables else "nav_lon"][:]
  if len(lat.shape) > 1: lat = lat[:,0]
  if len(lon.shape) > 1: lon = lon[0,:]
  pts = list()
  for xlat, xlon in IDx:
    ilat = np.where(xlat == lat)[0][0]
    ilon = np.where(xlon == lon)[0][0]
    if "land" in nc.dimensions:
      xland = ilat * len(lon) + ilon + 1
      iland = np.where(xland == land)[0][0]
    else: iland = None
    pts.append((ilat, ilon, iland))

  # building aligned forcing
  for year in range(varlist["climate"]["year_start"], varlist["climate"]["year_end"]+1):
    print("Building aligned forcing for year %s" % year)
    nc = Dataset(varlist["climate"]["sourcepath"] + varlist["climate"]["filename"] + str(year) + ".nc")
    ncout = Dataset(resultpath + "forcing_aligned_" + str(year) + ".nc", "w")
    for dim in nc.dimensions:
      if dim == "land": continue
      newdim = "lon" if dim in ["lon", "longitude", "x"] else "lat" if dim in ["lat", "latitude", "y"] else dim
      newsize = len(plat) if newdim == "lat" else len(plon) if newdim == "lon" else None if nc.dimensions[dim].isunlimited() else len(nc.dimensions[dim])
      ncout.createDimension(newdim, newsize)
    for var in nc.variables:
      print(var)
      if var == "land": continue
      elif len(nc.variables[var].dimensions) == 3 or "land" in nc.variables[var].dimensions: newdims = (nc.variables[var].dimensions[0], "lat", "lon")
      elif len(nc.variables[var].dimensions) == 2: newdims = ("lat", "lon")
      elif nc.variables[var].dimensions[0] in ["lon", "longitude", "x"]: newdims = ("lon",)
      elif nc.variables[var].dimensions[0] in ["lat", "latitude", "y"]: newdims = ("lat",)
      else: newdims = nc.variables[var].dimensions
      ncout.createVariable(var, nc.variables[var].dtype, newdims)
      ncout.variables[var].setncatts(nc.variables[var].__dict__)
      if len(nc.variables[var].dimensions) == 3:
        for idx, (ilat, ilon, iland) in enumerate(pts):
          ncout.variables[var][:, idx//nlon, idx%nlon] = nc.variables[var][:, ilat, ilon]
      elif nc.variables[var].dimensions[-1] == "land":
        for idx, (ilat, ilon, iland) in enumerate(pts):
          ncout.variables[var][:, idx//nlon, idx%nlon] = nc.variables[var][:, iland]
      elif var == "nav_lat": ncout.variables[var][:] = pnavlat
      elif var == "nav_lon": ncout.variables[var][:] = pnavlon
      elif var == "lat": ncout.variables[var][:] = plat
      elif var == "lon": ncout.variables[var][:] = plon
      elif var == "contfrac": ncout.variables[var][:] = 1
      else: ncout.variables[var][:] = nc.variables[var][:]
    ncout.close()
    nc.close()

  # building aligned restart files
  for path in varlist["restart"]:
    print("Building aligned file for", path)
    nc = Dataset(path)
    ncout = Dataset(resultpath + os.path.split(path)[-1], "w")
    for dim in nc.dimensions:
      newsize = len(plat) if dim in ["lat", "latitude", "y"] else len(plon) if dim in ["lon", "longitude", "x"] else None if nc.dimensions[dim].isunlimited() else len(nc.dimensions[dim])
      ncout.createDimension(dim, newsize)
    for var in nc.variables:
      ncout.createVariable(var, nc.variables[var].dtype, nc.variables[var].dimensions)
      ncout.variables[var].setncatts(nc.variables[var].__dict__)
      if var in ["nav_lat", "lat", "LAT", "latitude", "y"]:
        if len(nc.variables[var].dimensions) == 2: ncout.variables[var][:] = pnavlat
        else: ncout.variables[var][:] = plat
      elif var in ["nav_lon", "lon", "LON", "longitude", "x"]:
        if len(nc.variables[var].dimensions) == 2: ncout.variables[var][:] = pnavlon
        else: ncout.variables[var][:] = plon
      elif nc.variables[var].dimensions[-1] in ["lon", "longitude", "x"] and nc.variables[var].dimensions[-2] in ["lat", "latitude", "y"]:
        for idx, (ilat, ilon, iland) in enumerate(pts):
          ncout.variables[var][..., idx//nlon, idx%nlon] = nc.variables[var][..., ilat, ilon]
      else:
        ncout.variables[var][:] = nc.variables[var][:]
    nc.close()
    ncout.close()
    
