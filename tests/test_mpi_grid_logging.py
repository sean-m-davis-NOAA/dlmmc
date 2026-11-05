import logging
import sys
import types

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("netCDF4")

import dlmmc.mpi as dlmmc_mpi


class _SingleRankComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, value, root=0):
        return value

    def Barrier(self):
        pass


@pytest.fixture
def single_rank_mpi(monkeypatch):
    mpi = types.SimpleNamespace(COMM_WORLD=_SingleRankComm())
    package = types.ModuleType("mpi4py")
    package.MPI = mpi
    monkeypatch.setitem(sys.modules, "mpi4py", package)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)
    monkeypatch.setattr(dlmmc_mpi, "_get_compiled_model", lambda *args, **kwargs: None)


def test_fit_grid_mpi_logs_fit_time_for_attempted_cells_only(single_rank_mpi, monkeypatch, tmp_path, caplog):
    def failing_fit(*args, **kwargs):
        raise RuntimeError("sampler failed")

    monkeypatch.setattr(dlmmc_mpi, "fit", failing_fit)
    values = np.ones((6, 1, 2))
    values[:, 0, 0] = np.nan
    data = xr.DataArray(values, dims=("time", "pressure", "latitude"))

    with caplog.at_level(logging.INFO, logger="dlmmc.mpi"):
        result = dlmmc_mpi.fit_grid_mpi(
            data, tmp_path / "grid.nc", model="noregs_ar1", chains=1, samples=2
        )

    try:
        assert result["fit_status"].values.tolist() == [[0, -1]]
    finally:
        result.close()
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert messages[0].startswith("cell flat_index=1 rank=0 status=failed fit_seconds=")
