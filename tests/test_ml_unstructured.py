"""
Unit tests for Step 4 (ml.py) support of pseudo-unstructured source files.

These tests exercise:
  - detect_grid_type: structured, pseudo-unstructured (y × x=1), and true
    unstructured (cell dimension) grids.
  - _build_cell_idx_map: correct mapping from training-pixel global-grid indices
    to cell positions in a pseudo-unstructured source file.
  - extract_data: correct extraction of the response variable when the source
    file is in pseudo-unstructured format.
"""

import os
import tempfile

import numpy as np
import pytest

# Tools must be importable
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from netCDF4 import Dataset
from Tools.ml import _build_cell_idx_map, detect_grid_type, extract_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_structured_nc(path, nlat=5, nlon=6):
    """Create a minimal structured (lat/lon) netCDF file."""
    with Dataset(path, "w") as nc:
        nc.createDimension("y", nlat)
        nc.createDimension("x", nlon)
        nav_lat = nc.createVariable("nav_lat", "f4", ("y", "x"))
        nav_lon = nc.createVariable("nav_lon", "f4", ("y", "x"))
        lats = np.linspace(80, -80, nlat)
        lons = np.linspace(-150, 150, nlon)
        lon2d, lat2d = np.meshgrid(lons, lats)
        nav_lat[:] = lat2d
        nav_lon[:] = lon2d


def _make_pseudo_unstructured_nc(path, cell_lats, cell_lons, values=None):
    """
    Create a pseudo-unstructured netCDF file (y × x=1 layout) with nav_lat/
    nav_lon and a 2-D variable 'myvar' of shape (n_cells, 1).
    """
    n_cells = len(cell_lats)
    with Dataset(path, "w") as nc:
        nc.createDimension("y", n_cells)
        nc.createDimension("x", 1)
        nav_lat = nc.createVariable("nav_lat", "f4", ("y", "x"))
        nav_lon = nc.createVariable("nav_lon", "f4", ("y", "x"))
        myvar = nc.createVariable("myvar", "f4", ("y", "x"))
        nav_lat[:, 0] = cell_lats
        nav_lon[:, 0] = cell_lons
        if values is None:
            values = np.arange(n_cells, dtype=np.float32)
        myvar[:, 0] = values


def _make_cell_nc(path, n_cells=4):
    """Create a true unstructured netCDF file with a 'cell' dimension."""
    with Dataset(path, "w") as nc:
        nc.createDimension("cell", n_cells)
        lat = nc.createVariable("lat", "f4", ("cell",))
        lon = nc.createVariable("lon", "f4", ("cell",))
        lat[:] = np.array([10, 20, 30, 40], dtype=np.float32)[:n_cells]
        lon[:] = np.array([0, 10, 20, 30], dtype=np.float32)[:n_cells]


def _make_packdata(nlat, nlon, lat_reso, lon_reso, nlats, nlons):
    """
    Return a minimal xarray.Dataset that mimics the packdata used in ml_loop,
    carrying the Nlat/Nlon/lat_reso/lon_reso attributes.
    """
    import xarray as xr

    ds = xr.Dataset()
    ds.attrs.update(
        nlat=nlat,
        nlon=nlon,
        lat_reso=lat_reso,
        lon_reso=lon_reso,
        Nlat=np.array(nlats, dtype=int),
        Nlon=np.array(nlons, dtype=int),
    )
    return ds


# ---------------------------------------------------------------------------
# detect_grid_type
# ---------------------------------------------------------------------------


class TestDetectGridType:
    def test_structured(self, tmp_path):
        p = str(tmp_path / "structured.nc")
        _make_structured_nc(p, nlat=5, nlon=6)
        assert detect_grid_type(p) == "structured"

    def test_pseudo_unstructured(self, tmp_path):
        p = str(tmp_path / "pseudo.nc")
        _make_pseudo_unstructured_nc(p, [10.0, 20.0, 30.0], [0.0, 10.0, 20.0])
        assert detect_grid_type(p) == "unstructured"

    def test_true_unstructured_cell_dim(self, tmp_path):
        p = str(tmp_path / "cell.nc")
        _make_cell_nc(p)
        assert detect_grid_type(p) == "unstructured"

    def test_structured_multi_x(self, tmp_path):
        """A file with x > 1 must be detected as structured."""
        p = str(tmp_path / "struct2.nc")
        _make_structured_nc(p, nlat=4, nlon=3)
        assert detect_grid_type(p) == "structured"


# ---------------------------------------------------------------------------
# _build_cell_idx_map
# ---------------------------------------------------------------------------


class TestBuildCellIdxMap:
    """Tests for the mapping helper."""

    def _make_scenario(self, lat_reso, lon_reso, cell_lats, cell_lons, nlat, nlon):
        """
        Return (tmp_path, packdata, sourcefile_path) for a scenario where
        the source file contains *cell_lats/cell_lons* and the packdata
        Nlat/Nlon contain the same pixels expressed as global-grid indices.
        """
        # Convert cell_lats/lons to global-grid indices
        nlats = np.trunc((90 - np.array(cell_lats)) / lat_reso).astype(int)
        nlons = np.trunc((180 + np.array(cell_lons)) / lon_reso).astype(int)

        packdata = _make_packdata(nlat, nlon, lat_reso, lon_reso, nlats, nlons)
        return packdata, nlats, nlons

    def test_identity_mapping(self, tmp_path):
        """When every training pixel appears exactly once, expect identity mapping."""
        lat_reso, lon_reso = 2.0, 2.0
        cell_lats = [80.0, 60.0, 40.0, 20.0]
        cell_lons = [0.0, 10.0, 20.0, 30.0]

        p = str(tmp_path / "source.nc")
        _make_pseudo_unstructured_nc(p, cell_lats, cell_lons)

        packdata, nlats, nlons = self._make_scenario(
            lat_reso, lon_reso, cell_lats, cell_lons, 90, 180
        )

        cell_idx = _build_cell_idx_map(p, packdata)
        # Each training pixel should map to the corresponding cell in order
        assert list(cell_idx) == list(range(len(cell_lats)))

    def test_reordered_cells(self, tmp_path):
        """Training pixels in a different order than source cells."""
        lat_reso, lon_reso = 2.0, 2.0
        # Source file stores cells in this order
        cell_lats = [80.0, 60.0, 40.0, 20.0]
        cell_lons = [0.0, 10.0, 20.0, 30.0]

        p = str(tmp_path / "source.nc")
        _make_pseudo_unstructured_nc(p, cell_lats, cell_lons)

        # Training pixels reference them in reversed order
        train_lats = [20.0, 40.0, 60.0, 80.0]
        train_lons = [30.0, 20.0, 10.0, 0.0]
        packdata, _, _ = self._make_scenario(
            lat_reso, lon_reso, train_lats, train_lons, 90, 180
        )

        cell_idx = _build_cell_idx_map(p, packdata)
        # reversed order -> indices should be [3, 2, 1, 0]
        assert list(cell_idx) == [3, 2, 1, 0]

    def test_duplicate_training_pixels(self, tmp_path):
        """A training pixel may appear multiple times; each should resolve correctly."""
        lat_reso, lon_reso = 2.0, 2.0
        cell_lats = [80.0, 60.0, 40.0]
        cell_lons = [0.0, 10.0, 20.0]

        p = str(tmp_path / "source.nc")
        _make_pseudo_unstructured_nc(p, cell_lats, cell_lons)

        # Training set has pixel 1 repeated
        train_lats = [80.0, 60.0, 80.0]
        train_lons = [0.0, 10.0, 0.0]
        packdata, _, _ = self._make_scenario(
            lat_reso, lon_reso, train_lats, train_lons, 90, 180
        )

        cell_idx = _build_cell_idx_map(p, packdata)
        assert list(cell_idx) == [0, 1, 0]

    def test_missing_pixel_raises(self, tmp_path):
        """A training pixel absent from the source file must raise RuntimeError."""
        lat_reso, lon_reso = 2.0, 2.0
        cell_lats = [80.0, 60.0]
        cell_lons = [0.0, 10.0]

        p = str(tmp_path / "source.nc")
        _make_pseudo_unstructured_nc(p, cell_lats, cell_lons)

        # Training pixel at (40°N, 20°E) is NOT in the source file
        train_lats = [40.0]
        train_lons = [20.0]
        packdata, _, _ = self._make_scenario(
            lat_reso, lon_reso, train_lats, train_lons, 90, 180
        )

        with pytest.raises(RuntimeError, match="not found"):
            _build_cell_idx_map(p, packdata)

    def test_no_coordinate_vars_raises(self, tmp_path):
        """Source file without lat/lon variables must raise RuntimeError."""
        p = str(tmp_path / "no_coords.nc")
        with Dataset(p, "w") as nc:
            nc.createDimension("y", 3)
            nc.createDimension("x", 1)
            v = nc.createVariable("myvar", "f4", ("y", "x"))
            v[:] = 0

        packdata = _make_packdata(90, 180, 2.0, 2.0, [5], [10])
        with pytest.raises(RuntimeError, match="lat/lon coordinate variables"):
            _build_cell_idx_map(p, packdata)


# ---------------------------------------------------------------------------
# extract_data (unstructured branch)
# ---------------------------------------------------------------------------


class TestExtractDataUnstructured:
    """
    Integration-style test for extract_data when the source is in
    pseudo-unstructured format.

    We build synthetic packdata + ivar arrays and check that pool_arr is
    filled with the expected values.
    """

    def _make_minimal_packdata(self, nlats, nlons, cell_idx):
        """packdata with just enough attrs and one predictor variable."""
        import xarray as xr

        n_sel = len(nlats)
        nlat, nlon = 90, 180
        lat_reso, lon_reso = 2.0, 2.0

        # Single predictor variable (no veget dimension) of shape (nlat, nlon)
        pred = np.zeros((nlat, nlon), dtype=np.float32)

        ds = xr.Dataset({"SWdown_mean": (["lat", "lon"], pred)})
        ds.attrs.update(
            nlat=nlat,
            nlon=nlon,
            lat_reso=lat_reso,
            lon_reso=lon_reso,
            Nlat=np.array(nlats, dtype=int),
            Nlon=np.array(nlons, dtype=int),
            cell_idx=np.array(cell_idx, dtype=int),
        )
        return ds

    def test_unstructured_extraction(self):
        """
        extract_data should pick the correct values from a 1-D pool_map (n_cells,)
        using packdata.cell_idx.
        """
        # Source has 4 cells, values 10, 20, 30, 40
        n_cells = 4
        pool_vals = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        # Fake ivar: shape (npft=2, n_cells, x=1).  After np.squeeze → (npft, n_cells).
        # With ind=(1,): pool_map = squeezed[0] → shape (n_cells,).
        npft = 2
        ivar = np.zeros((npft, n_cells, 1), dtype=np.float64)
        ivar[0, :, 0] = pool_vals  # PFT 1 (index 0)

        # Training set picks cells [3, 0, 2] (i.e., values 40, 10, 30)
        cell_idx = [3, 0, 2]
        # Corresponding global grid indices (arbitrary, as long as consistent)
        nlats = [5, 10, 15]
        nlons = [10, 20, 30]

        packdata = self._make_minimal_packdata(nlats, nlons, cell_idx)

        varlist = {
            "resp": {"format": "unstructured"},
            "pred": {"allname_pft": []},
            "PFTmask": {"pred_thres": 0.1},
        }
        labx = ["Y", "SWdown_mean", "pft"]

        # We need a real PFT_mask_lai that extract_x.pft can use
        # (it does PFT_mask_lai[ipft-1, Nlat, Nlon]).
        # Provide a 3-D mask with shape (1, 90, 180).
        PFT_mask_lai = np.zeros((1, 90, 180), dtype=np.float32)
        for ilat, ilon in zip(nlats, nlons):
            PFT_mask_lai[0, ilat, ilon] = 1.0

        df_data, pool_map = extract_data(
            packdata,
            ivar,
            ipft=1,
            PFT_mask_lai=PFT_mask_lai,
            varlist=varlist,
            labx=labx,
            ind=(1,),
        )

        expected_Y = pool_vals[cell_idx]  # [40, 10, 30]
        actual_Y = df_data["Y"].values[: len(cell_idx)]
        np.testing.assert_array_almost_equal(actual_Y, expected_Y)

        # pool_map must be 2D (nlat, nlon) for mleval.evaluation_map to work
        assert pool_map.shape == (90, 180), (
            f"pool_map must be 2D (nlat, nlon), got shape {pool_map.shape}"
        )

        # Training pixel locations should have the correct values
        for j, (ilat, ilon) in enumerate(zip(nlats, nlons)):
            expected_val = pool_vals[cell_idx[j]]
            np.testing.assert_almost_equal(
                pool_map[ilat, ilon],
                expected_val,
                err_msg=f"pool_map[{ilat},{ilon}] should be {expected_val}",
            )

        # Non-training pixels should be NaN
        n_valid = np.sum(~np.isnan(pool_map))
        assert n_valid == len(cell_idx), (
            f"Expected {len(cell_idx)} non-NaN pixels, got {n_valid}"
        )
