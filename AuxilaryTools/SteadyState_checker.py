# Author: Daniel Goll dsgoll123@gmail.com
# Date: 2023/01/05
# Description: This script performs analysis regarding the equilibration of the land carbon cycling in ORCHIDEE simulations.
# The data is read in from ORCHIDEE output files and stored in numpy arrays.
# The script then performs a linear trend analysis and writes it to a text file.
# The script saves the C cycle information analyzed in netCDF files.



import numpy as np
from mysimulations_specs import *
from netCDF4 import Dataset
from scipy.stats import linregress

# text file which stores information on the analysis
logfile = expid + which_run + "_info_pixles_at_steadystate"


def check(string, filename):
    # Open the file in append mode
    with open(filename, "a") as file:
        # Write to the file with line break added:
        file.write(string + "\n")


check(expid, logfile)
check(which_run, logfile)

# parameters:
ntime = len(decades) * 10
missval = -99999
varnames = ["Ctot"]
thr = 1.0
cov_thres = 0.1


adict = locals()

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for varname in varnames:
    adict[varname] = np.ma.zeros((ntime, npft, nlat, nlon))

##########################################################
cnt = 0
for idec in range(len(decades)):
    # model output file
    fn1_raw = (
        igcm_out
        + "/"
        + expid
        + which_run
        + "/SBG/Output/MO/"
        + expid
        + which_run
        + "_"
        + str(decades[idec])
        + "0101_"
        + str(decades[idec] + 9)
        + "1231_1M_stomate_history.nc"
    )
    # fn1_raw=igcm_out+'/'+expid+which_run+'/SBG/Output/YE/'+expid+which_run+'_'+str(decades[idec])+'0101_'+str(decades[idec]+9)+'1231_1Y_stomate_history.nc'
    ncres1 = Dataset(fn1_raw, "r")
    veg = ncres1.variables["VEGET_COV_MAX"][:]
    for varname in varnames:
        if varname == "Cveg":
            tmp = ncres1.variables["TOTAL_M_c"][:]
            tmp2 = tmp
        if varname == "Ctot":
            tmpa = (
                ncres1.variables["SOIL_SOMA_c"][:]
                + ncres1.variables["SOIL_SOMC_c"][:]
                + ncres1.variables["SOIL_SOMP_c"][:]
            )
            tmpb = (
                ncres1.variables["MICROBE_R_AB_c"][:]
                + ncres1.variables["MICROBE_K_AB_c"][:]
                + ncres1.variables["MICROBE_R_BE_c"][:]
                + ncres1.variables["MICROBE_K_BE_c"][:]
            )
            tmpc = ncres1.variables["TOTAL_M_c"][:]
            tmpd = (
                ncres1.variables["LITTER_STR_AB_c"][:]
                + ncres1.variables["LITTER_STR_BE_c"][:]
                + ncres1.variables["LITTER_MET_AB_c"][:]
                + ncres1.variables["LITTER_MET_BE_c"][:]
                + ncres1.variables["LITTER_WOD_AB_c"][:]
                + ncres1.variables["LITTER_WOD_BE_c"][:]
            )
            tmp = tmpa + tmpb + tmpc + tmpd
            tmp2 = tmp

        if varname == "Ctot2":
            tmpa = (
                ncres1.variables["CARBON_PASSIVE"][:]
                + ncres1.variables["CARBON_ACTIVE"][:]
                + ncres1.variables["CARBON_SLOW"][:]
            )
            tmpc = ncres1.variables["TOTAL_M"][:]
            tmpd = (
                ncres1.variables["LITTER_STR_AB"][:]
                + ncres1.variables["LITTER_STR_BE"][:]
                + ncres1.variables["LITTER_MET_AB"][:]
                + ncres1.variables["LITTER_MET_BE"][:]
            )
            tmp = tmpa + tmpc + tmpd
            tmp2 = tmp

        if varname == "Csoil":
            tmpa = (
                ncres1.variables["SOIL_SOMA_c"][:]
                + ncres1.variables["SOIL_SOMC_c"][:]
                + ncres1.variables["SOIL_SOMP_c"][:]
            )
            tmpb = (
                ncres1.variables["MICROBE_R_AB_c"][:]
                + ncres1.variables["MICROBE_K_AB_c"][:]
                + ncres1.variables["MICROBE_R_BE_c"][:]
                + ncres1.variables["MICROBE_K_BE_c"][:]
            )
            tmp = tmpa + tmpb
            tmp2 = tmp

        ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tmp2[veg < cov_thres] = missval
        tmp2 = tmp2
        tmp2[tmp2.mask] = missval

        yr1 = idec + cnt
        adict[varname][yr1 : yr1 + 10] = tmp2
        cnt = cnt + 9
    ncres1.close()

print("read done")

# compute number of pixels with drift below threshold
check(
    "simulation years: " + str(decades[0]) + "-" + str(decades[len(decades) - 1] + 9),
    logfile,
)
check("cover fraction threshold=" + str(cov_thres), logfile)
check("EQ threshold=" + str(thr) + " g m-2 yr-1", logfile)
check("n years=" + str(ntime), logfile)

for varname in varnames:
    cnt = 0
    A_za = 0.0
    A_total = 0.0
    for pft in range(npft):
        X = adict[varname][:, pft, :, :]
        masked_X = np.ma.masked_array(X, mask=np.where(X == missval, True, False))

        # Create a 2D array to store the trend values
        trend = np.empty((nlat, nlon))

        # Loop over the latitude and longitude dimensions
        for i in range(nlat):
            for j in range(nlon):
                # Extract the time series at each location
                time_series = masked_X[:, i, j]

                # Get the non-masked values using the compressed method
                time_series_compressed = time_series.compressed()

                # Fit a linear regression model to the time series using linregress
                if time_series_compressed.size > 0:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        np.arange(ntime), time_series_compressed
                    )
                    trend[i, j] = slope

                else:
                    trend[i, j] = missval

                # Store the slope value in the trend array (the slope is the linear trend at this
                # location)

        # count entries in trend above threshold:
        if np.all(trend == missval):
            print(
                (
                    varname
                    + " for PFT"
                    + str(pft + 1)
                    + ": no pixels with coverage above:"
                    + str(cov_thres)
                )
            )
            total = 0.0
            za = 0.0
        else:
            masked_trend = np.ma.masked_array(
                trend, mask=np.where(trend == missval, True, False)
            )

            p31 = np.asarray(masked_trend[:, :])
            za = (abs(p31) < thr).sum()
            # count total pixels :
            total = np.ma.count(masked_trend)
            zr = za / float(total) * 100.0
            check(
                varname
                + " for PFT"
                + str(pft + 1)
                + ": percentage of pixel below threshold: "
                + str(round(zr, 2))
                + "% of "
                + str(total)
                + " pixels",
                logfile,
            )

        if pft > 0:
            A_total = A_total + total
            A_za = A_za + za

    A_zr = A_za / float(A_total) * 100.0
    check(
        varname
        + " GLOBAL (excl PFT1): percentage of tiles below threshold: "
        + str(round(A_zr, 2))
        + "% of "
        + str(A_total)
        + " tiles",
        logfile,
    )
    # print(varname+' for PFT'+str(pft+1)+': 5th,50th,95th percentiles:'+str(np.nanpercentile(trend,5))+' / '+str(np.nanpercentile(trend,50))+' / '+str(np.nanpercentile(trend, 95)))


print("analysis done")

##########################################################
for varname in varnames:
    for pft in range(npft):
        # output folder:
        file2create = (
            trendy_out
            + "/ORCHIDEE-CNP_"
            + which_run
            + "_"
            + varname
            + "_PFT"
            + str(pft + 1)
            + ".nc"
        )
        nc = Dataset(file2create, "w")
        ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        nc.createDimension("lon", nlon)
        nc.createDimension("lat", nlat)
        nc.createDimension("time", ntime)

        var = nc.createVariable("time", "f", ("time",))

        tmp = np.zeros((ntime))
        for loop in range(ntime):
            tmp[loop] = (loop) * 365 + 365
        var[:] = tmp * 1.0  # NOTE

        tmp_str = "days since " + str(decades[0]) + "-01-01 00:00:00"
        nc.variables["time"].__setattr__("units", tmp_str)
        nc.variables["time"].__setattr__("long_name", "time")
        nc.variables["time"].__setattr__("calendar", "noleap")
        #################################################################
        var = nc.createVariable(varname, "f", ("time", "lat", "lon"), fill_value=-99999)
        ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        var[:] = adict[varname][:, pft, :, :]
        ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if varname == "Cveg":
            nc.variables[varname].__setattr__("units", "g m-2")
            nc.variables[varname].__setattr__("long_name", "Total biomass carbon")
        if varname == "Ctot":
            nc.variables[varname].__setattr__("units", "g m-2")
            nc.variables[varname].__setattr__("long_name", "Total organic carbon")
        #################################################################
        nc.close()

print("write done")
