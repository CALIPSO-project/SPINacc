from Tools import *

def write(varlist,resultpath,IDx):
  climvar=varlist['climate']
  for year in range(climvar['year_start'], climvar['year_end']+1):
    nc = Dataset(climvar['sourcepath']+climvar['filename']+str(year)+'.nc')
    if 'land' in nc.dimensions: land = nc.variables['land'][:]
    lat = nc.variables['lat' if 'lat' in nc.variables else 'latitude' if 'latitude' in nc.variables else 'nav_lat'][:]
    lon = nc.variables['lon' if 'lon' in nc.variables else 'longitude' if 'longitude' in nc.variables else 'nav_lon'][:]
    if len(lat.shape) > 1: lat = lat[:,0]
    if len(lon.shape) > 1: lon = lon[0,:]
    pts = list()
    for xlat, xlon in IDx:
      ilat = np.where(xlat == lat)[0][0]
      ilon = np.where(xlon == lon)[0][0]
      pts.append(ilat * len(lon) + ilon + 1)
    print("Building compressed forcing for year %s" % year)
    ncout = Dataset(resultpath+'forcing_compressed_'+str(year)+'.nc', 'w')
    for dim in nc.dimensions:
      newdim = "x" if dim in ["lon", "longitude"] else "y" if dim in ["lat", "latitude"] else dim
      newsize = len(pts) if dim == "land" else None if nc.dimensions[dim].isunlimited() else len(nc.dimensions[dim])
      ncout.createDimension(newdim, newsize)
    if "land" not in ncout.dimensions:
      ncout.createDimension("land", len(pts))
      ncout.createVariable("land", "i4", ("land",))
      ncout.variables["land"].setncattr("compress", "y x")
      ncout.variables["land"][:] = pts
    for var in nc.variables:
      if len(nc.variables[var].dimensions) == 3: newdims = (nc.variables[var].dimensions[0], "land")
      elif "land" not in nc.dimensions and len(nc.variables[var].dimensions) == 2: newdims = ("y", "x")
      else: newdims = nc.variables[var].dimensions
      ncout.createVariable(var, nc.variables[var].dtype, newdims)
      ncout.variables[var].setncatts(nc.variables[var].__dict__)
      if var == "land": ncout.variables[var][:] = pts
      elif len(nc.variables[var].dimensions) == 3:
        for idx, pt in enumerate(pts):
          ilat = (pt - 1) // len(lon)
          ilon = (pt - 1) % len(lon)
          ncout.variables[var][:, idx] = nc.variables[var][:, ilat, ilon]
      elif nc.variables[var].dimensions[-1] == "land":
        for idx, pt in enumerate(pts):
          iland = np.where(pt == land)[0][0]
          ncout.variables[var][:, idx] = nc.variables[var][:, iland]
      else:
        ncout.variables[var][:] = nc.variables[var][:]
    ncout.close()
    nc.close()
