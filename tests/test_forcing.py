from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from Tools.forcing import write


def _write_daily_forcing(path: Path):
    with Dataset(path, "w") as nc:
        nc.createDimension("tstep", 365)
        nc.createDimension("latitude", 2)
        nc.createDimension("longitude", 2)

        time = nc.createVariable("time", "f8", ("tstep",))
        time[:] = np.arange(365)

        latitude = nc.createVariable("latitude", "f4", ("latitude",))
        latitude.units = "degrees_north"
        latitude[:] = [45.0, 43.0]

        longitude = nc.createVariable("longitude", "f4", ("longitude",))
        longitude.units = "degrees_east"
        longitude[:] = [5.0, 7.0]

        nav_lat = nc.createVariable("nav_lat", "f4", ("latitude", "longitude"))
        nav_lon = nc.createVariable("nav_lon", "f4", ("latitude", "longitude"))
        nav_lat[:] = [[45.0, 45.0], [43.0, 43.0]]
        nav_lon[:] = [[5.0, 7.0], [5.0, 7.0]]

        contfrac = nc.createVariable("contfrac", "f4", ("latitude", "longitude"))
        contfrac[:] = np.ones((2, 2), dtype=np.float32)

        tair = nc.createVariable("Tair", "f4", ("tstep", "latitude", "longitude"))
        tair[:] = np.broadcast_to(
            np.arange(365, dtype=np.float32).reshape(365, 1, 1), (365, 2, 2)
        )


def test_write_unstructured_forcing_supports_daily_latitude_longitude(tmp_path):
    _write_daily_forcing(tmp_path / "forcing_1901.nc")

    varlist = {
        "climate": {
            "sourcepath": str(tmp_path) + "/",
            "filename": "forcing_",
            "year_start": 1901,
            "year_end": 1901,
        },
        "resp": {"format": "unstructured"},
        "restart": [],
    }
    idx = np.array([(45.0, 5.0), (43.0, 7.0)])

    write(varlist, tmp_path, idx)

    with Dataset(tmp_path / "forcing_unstructured_1901.nc") as nc:
        assert nc.variables["lat"][:].tolist() == [45.0, 43.0]
        assert nc.variables["lon"][:].tolist() == [5.0, 7.0]
        assert nc.variables["Tair"].shape == (365, 2)
        assert nc.variables["contfrac"].shape == (2,)
        assert "latitude" not in nc.variables
        assert "longitude" not in nc.variables
