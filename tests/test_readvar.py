import calendar
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset

from Tools.readvar import readvar


MONTH_DAYS = np.array(calendar.mdays[1:])


def _write_coord_ref(path: Path):
    with Dataset(path, "w") as nc:
        nc.createDimension("y", 2)
        nc.createDimension("x", 2)
        nav_lat = nc.createVariable("nav_lat", "f4", ("y", "x"))
        nav_lon = nc.createVariable("nav_lon", "f4", ("y", "x"))
        nav_lat[:] = [[45.0, 45.0], [43.0, 43.0]]
        nav_lon[:] = [[5.0, 7.0], [5.0, 7.0]]


def _write_pft_mask(path: Path):
    with Dataset(path, "w") as nc:
        nc.createDimension("veget", 2)
        nc.createDimension("lat", 2)
        nc.createDimension("lon", 2)
        var = nc.createVariable("VEGET_COV_MAX", "f4", ("veget", "lat", "lon"))
        var[:] = np.ones((2, 2, 2), dtype=np.float32)


def _write_pred_source(path: Path):
    with Dataset(path, "w") as nc:
        nc.createDimension("veget", 2)
        nc.createDimension("lat", 2)
        nc.createDimension("lon", 2)
        npp = nc.createVariable("NPP", "f4", ("veget", "lat", "lon"))
        lai = nc.createVariable("LAI", "f4", ("veget", "lat", "lon"))
        npp[:] = np.full((2, 2, 2), 10.0, dtype=np.float32)
        lai[:] = np.full((2, 2, 2), 2.0, dtype=np.float32)


def _write_soil_source(path: Path):
    with Dataset(path, "w") as nc:
        nc.createDimension("lat", 2)
        nc.createDimension("lon", 2)
        clay = nc.createVariable("clay_frac", "f4", ("lat", "lon"))
        clay[:] = np.full((2, 2), 0.3, dtype=np.float32)


def _expand_monthly(monthly_values, steps_per_day):
    values = []
    for month, days in enumerate(MONTH_DAYS, start=1):
        month_value = monthly_values(month)
        for _ in range(days):
            for _ in range(steps_per_day):
                values.append(month_value)
    return np.array(values, dtype=np.float32).reshape(-1, 1, 1)


def _write_climate_file(path: Path, variables, steps_per_day):
    nsteps = int(MONTH_DAYS.sum() * steps_per_day)
    with Dataset(path, "w") as nc:
        nc.createDimension("tstep", nsteps)
        nc.createDimension("latitude", 2)
        nc.createDimension("longitude", 2)
        time = nc.createVariable("time", "f8", ("tstep",))
        time[:] = np.arange(nsteps)
        nc.createVariable("latitude", "f4", ("latitude",))[:] = [45.0, 43.0]
        nc.createVariable("longitude", "f4", ("longitude",))[:] = [5.0, 7.0]
        nc.createVariable("nav_lat", "f4", ("latitude", "longitude"))[:] = [
            [45.0, 45.0],
            [43.0, 43.0],
        ]
        nc.createVariable("nav_lon", "f4", ("latitude", "longitude"))[:] = [
            [5.0, 7.0],
            [5.0, 7.0],
        ]
        nc.createVariable("contfrac", "f4", ("latitude", "longitude"))[:] = np.ones(
            (2, 2), dtype=np.float32
        )
        for name, monthly_values in variables.items():
            var = nc.createVariable(
                name,
                "f4",
                ("tstep", "latitude", "longitude"),
                fill_value=np.float32(1.0e20),
            )
            data = _expand_monthly(monthly_values, steps_per_day)
            var[:] = np.broadcast_to(data, (nsteps, 2, 2))


def _make_varlist(tmp_path: Path, climate_variables):
    return {
        "coord_ref": str(tmp_path / "coord_ref.nc"),
        "lat_reso": 2,
        "lon_reso": 2,
        "climate": {
            "sourcepath": str(tmp_path) + "/",
            "filename": "forcing_",
            "year_start": 1901,
            "year_end": 1901,
            "variables": climate_variables,
        },
        "PFTmask": {
            "sourcefile": str(tmp_path / "pftmask.nc"),
            "var": "VEGET_COV_MAX",
            "cluster_thres": 0.001,
            "pred_thres": 0.1,
        },
        "pred": {
            "var1": {
                "sourcefile": str(tmp_path / "pred.nc"),
                "variables": ["NPP", "LAI"],
                "rename": ["NPP0", "LAI0"],
            },
            "var2": {
                "sourcefile": str(tmp_path / "soil.nc"),
                "variables": ["clay_frac"],
            },
            "clustering": [
                "Tamp",
                "Tmax",
                "Tmean",
                "Tmin",
                "Tstd",
                "precip_mean",
                "precip_std",
                "Wind_mean",
                "Qair_mean",
                "PSurf_mean",
                "SWdown_mean",
                "LWdown_mean",
                "clay_frac",
                "Pre_GS",
                "Temp_GS",
                "GS_length",
                "interx1",
                "interx2",
            ],
            "allname": [
                "Tamp",
                "Tmax",
                "Tmean",
                "Tmin",
                "Tstd",
                "precip_mean",
                "precip_std",
                "Wind_mean",
                "Qair_mean",
                "PSurf_mean",
                "SWdown_mean",
                "LWdown_mean",
                "clay_frac",
                "Pre_GS",
                "Temp_GS",
                "GS_length",
                "interx1",
                "interx2",
            ],
            "allname_pft": ["NPP0", "LAI0"],
        },
    }


def _config():
    return SimpleNamespace(
        take_year_average=False,
        max_kmeans_clusters=9,
        kmeans_clusters=4,
    )


def _scalar(ds, name):
    return ds[name].isel(lat=0, lon=0).item()


def test_readvar_supports_daily_tmax_tmin_and_scalar_wind(tmp_path):
    _write_coord_ref(tmp_path / "coord_ref.nc")
    _write_pft_mask(tmp_path / "pftmask.nc")
    _write_pred_source(tmp_path / "pred.nc")
    _write_soil_source(tmp_path / "soil.nc")
    _write_climate_file(
        tmp_path / "forcing_1901.nc",
        {
            "Tmax": lambda month: 280.0 + month,
            "Tmin": lambda month: 270.0 + month,
            "Qair": lambda month: 0.001 * month,
            "PSurf": lambda month: 100000.0 + month,
            "Wind": lambda month: 1.0 + month,
            "precip": lambda month: 0.001 * month,
            "Rainf": lambda month: 0.0008 * month,
            "Snowf": lambda month: 0.0002 * month,
            "LWdown": lambda month: 200.0 + month,
            "SWdown": lambda month: 100.0 + month,
        },
        steps_per_day=1,
    )

    ds = readvar(
        _make_varlist(
            tmp_path,
            [
                "Tmax",
                "Tmin",
                "Qair",
                "PSurf",
                "Wind",
                "precip",
                "Rainf",
                "Snowf",
                "LWdown",
                "SWdown",
            ],
        ),
        _config(),
        None,
    )

    assert np.isclose(_scalar(ds, "Tmean"), 8.35)
    assert np.isclose(_scalar(ds, "Tmin"), 2.85)
    assert np.isclose(_scalar(ds, "Tmax"), 13.85)
    assert np.isclose(_scalar(ds, "Tamp"), 11.0)
    assert np.isclose(_scalar(ds, "Wind_mean"), 7.5)
    assert np.isclose(_scalar(ds, "precip_mean"), 204984.0)
    assert np.isclose(_scalar(ds, "Pre_GS"), 202176.0)
    assert np.isclose(
        _scalar(ds, "interx1"), _scalar(ds, "Tmean") * _scalar(ds, "precip_mean")
    )


def test_readvar_supports_legacy_6hourly_tair_and_wind_components(tmp_path):
    _write_coord_ref(tmp_path / "coord_ref.nc")
    _write_pft_mask(tmp_path / "pftmask.nc")
    _write_pred_source(tmp_path / "pred.nc")
    _write_soil_source(tmp_path / "soil.nc")
    _write_climate_file(
        tmp_path / "forcing_1901.nc",
        {
            "Tair": lambda month: 270.0 + month,
            "Qair": lambda month: 0.001 * month,
            "PSurf": lambda month: 100000.0 + month,
            "Wind_E": lambda month: 3.0,
            "Wind_N": lambda month: 4.0,
            "Rainf": lambda month: 0.0005 * month,
            "Snowf": lambda month: 0.0001 * month,
            "LWdown": lambda month: 200.0 + month,
            "SWdown": lambda month: 100.0 + month,
        },
        steps_per_day=4,
    )

    ds = readvar(
        _make_varlist(
            tmp_path,
            [
                "Tair",
                "Qair",
                "PSurf",
                "Wind_E",
                "Wind_N",
                "Rainf",
                "Snowf",
                "LWdown",
                "SWdown",
            ],
        ),
        _config(),
        None,
    )

    assert np.isclose(_scalar(ds, "Tmean"), 3.35)
    assert np.isclose(_scalar(ds, "precip_mean"), 122990.4)
    assert np.isclose(_scalar(ds, "Wind_mean"), 5.0)
    assert np.isclose(
        _scalar(ds, "interx1"), _scalar(ds, "Tmean") * _scalar(ds, "precip_mean")
    )
