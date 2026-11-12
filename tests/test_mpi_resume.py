import sys
import types

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("netCDF4")

import dlmmc.mpi as dlmmc_mpi
from dlmmc.mpi import estimate_grid_run, format_grid_estimate


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


class _WallClockLimit(BaseException):
    """Stands in for Slurm killing the job; not caught as a fit failure."""


def _data():
    values = np.ones((6, 1, 4))
    values[:, 0, 1] = 2
    values[:, 0, 2] = np.nan
    values[:, 0, 3] = 4
    return xr.DataArray(
        values,
        dims=("time", "pressure", "latitude"),
        coords={"pressure": [10.0], "latitude": [-30, -10, 10, 30]},
    )


def _posterior(value):
    posterior = xr.Dataset({
        name: (("chain", "draw"), np.full((2, 3), value))
        for name in ("sigma_trend", "sigma_AR", "rhoAR1", "sigma_seas")
    })
    posterior.update({
        name: (("chain", "draw", "time"), np.full((2, 3, 6), value))
        for name in ("trend", "slope", "ar", "seasonal")
    })
    posterior.attrs["divergent_transitions"] = 0
    return posterior


def _fit_grid(path, **kwargs):
    options = dict(model="noregs_ar1", chains=2, samples=3, seed=5, output_variables={"sigma_AR": "single"})
    options.update(kwargs)
    return dlmmc_mpi.fit_grid_mpi(_data(), path, **options)


def test_resume_fits_only_incomplete_cells_after_interruption(single_rank_mpi, monkeypatch, tmp_path):
    path = tmp_path / "grid.nc"

    def interrupted_fit(series, **kwargs):
        if series.values[0] == 2:
            raise _WallClockLimit
        return _posterior(series.values[0])

    monkeypatch.setattr(dlmmc_mpi, "fit", interrupted_fit)
    with pytest.raises(_WallClockLimit):
        _fit_grid(path)
    with xr.open_dataset(path) as partial:
        np.testing.assert_array_equal(partial["fit_status"], [[1, 0, 0, 0]])

    fitted = []

    def resumed_fit(series, **kwargs):
        fitted.append(float(series.values[0]))
        return _posterior(series.values[0])

    monkeypatch.setattr(dlmmc_mpi, "fit", resumed_fit)
    result = _fit_grid(path, resume=True)
    try:
        assert fitted == [2.0, 4.0]
        np.testing.assert_array_equal(result["fit_status"], [[1, 1, 0, 1]])
        np.testing.assert_array_equal(result["sigma_AR_mean"], [[1, 2, np.nan, 4]])
        assert "resumed: 1 cells complete, 2 to fit" in result.attrs["history"]
    finally:
        result.close()


def test_resume_retries_failed_cells_and_skips_complete_grid(single_rank_mpi, monkeypatch, tmp_path):
    path = tmp_path / "grid.nc"

    def failing_fit(series, **kwargs):
        if series.values[0] == 4:
            raise RuntimeError("sampler failed")
        return _posterior(series.values[0])

    monkeypatch.setattr(dlmmc_mpi, "fit", failing_fit)
    _fit_grid(path).close()

    fitted = []
    monkeypatch.setattr(dlmmc_mpi, "fit", lambda series, **kwargs: fitted.append(series.values[0]) or _posterior(1))
    _fit_grid(path, resume=True).close()
    assert fitted == [4.0]
    result = _fit_grid(path, resume=True)
    try:
        assert fitted == [4.0]
        np.testing.assert_array_equal(result["fit_status"], [[1, 1, 0, 1]])
        assert result.attrs["history"].count("resumed") == 2
    finally:
        result.close()


def test_resume_without_existing_file_starts_new_run(single_rank_mpi, monkeypatch, tmp_path):
    monkeypatch.setattr(dlmmc_mpi, "fit", lambda series, **kwargs: _posterior(1))
    result = _fit_grid(tmp_path / "new.nc", resume=True)
    try:
        np.testing.assert_array_equal(result["fit_status"], [[1, 1, 0, 1]])
        assert "history" not in result.attrs
    finally:
        result.close()


def test_resume_accepts_legacy_cross_cell_chunks(single_rank_mpi, monkeypatch, tmp_path):
    path = tmp_path / "legacy.nc"
    monkeypatch.setattr(dlmmc_mpi, "fit", lambda series, **kwargs: _posterior(1))
    with monkeypatch.context() as legacy:
        legacy.setattr(
            dlmmc_mpi, "_cell_chunks",
            lambda dims, sizes, grid_dims, dtype: tuple(sizes[dim] for dim in dims),
        )
        _fit_grid(path, max_cells=1).close()

    result = _fit_grid(path, resume=True)
    try:
        np.testing.assert_array_equal(result["fit_status"], [[1, 1, 0, 1]])
    finally:
        result.close()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"samples": 4}, "dimension 'draw' is 3, expected 4"),
        ({"seed": 6}, "setting 'seed'"),
        ({"adapt_delta": 0.95}, "setting 'adapt_delta' is only set in the run"),
        ({"sigma_AR_prior": 0.2}, "setting 'prior_sigma_AR'"),
        ({"output_variables": {"sigma_AR": "double"}}, "output variables differ"),
        ({"posterior_means": False}, "output variables differ"),
    ],
)
def test_resume_rejects_different_settings_and_keeps_file(single_rank_mpi, monkeypatch, tmp_path, change, message):
    path = tmp_path / "grid.nc"
    monkeypatch.setattr(dlmmc_mpi, "fit", lambda series, **kwargs: _posterior(1))
    _fit_grid(path).close()
    before = path.read_bytes()
    monkeypatch.setattr(dlmmc_mpi, "fit", lambda *args, **kwargs: pytest.fail("must not fit"))

    with pytest.raises(RuntimeError, match="Cannot resume") as error:
        _fit_grid(path, resume=True, **change)

    assert message in str(error.value)
    assert path.read_bytes() == before


def test_existing_output_requires_resume_or_overwrite(single_rank_mpi, monkeypatch, tmp_path):
    path = tmp_path / "grid.nc"
    monkeypatch.setattr(dlmmc_mpi, "fit", lambda series, **kwargs: _posterior(1))
    _fit_grid(path).close()

    with pytest.raises(RuntimeError, match="set resume=True to continue it"):
        _fit_grid(path)
    with pytest.raises(RuntimeError, match="only one of overwrite=True and resume=True"):
        _fit_grid(path, overwrite=True, resume=True)


def test_estimate_excludes_completed_cells(single_rank_mpi, monkeypatch, tmp_path):
    path = tmp_path / "grid.nc"
    monkeypatch.setattr(
        dlmmc_mpi, "fit",
        lambda series, **kwargs: (_ for _ in ()).throw(_WallClockLimit) if series.values[0] == 4 else _posterior(1),
    )
    with pytest.raises(_WallClockLimit):
        _fit_grid(path)

    estimate = estimate_grid_run(_data(), ranks=3, model="noregs_ar1", grid_dims=("pressure", "latitude"),
                                 resume_from=path)
    fresh = estimate_grid_run(_data(), ranks=3, model="noregs_ar1", resume_from=tmp_path / "missing.nc")

    assert (estimate["fitted_cells"], estimate["completed_cells"], estimate["skipped_cells"]) == (1, 2, 1)
    assert (estimate["max_cells_per_worker"], estimate["idle_workers"]) == (1, 1)
    assert "Cells to fit: 1 of 4 (2 already complete, 1 skipped" in format_grid_estimate(estimate)
    assert (fresh["fitted_cells"], fresh["completed_cells"]) == (3, 0)
    assert "already complete" not in format_grid_estimate(fresh)


def test_runner_forwards_resume(monkeypatch, tmp_path):
    from pathlib import Path

    import dlm_lat_alt_mpi_run

    captured = {}
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(
        dlm_lat_alt_mpi_run.dlmmc, "fit_grid_mpi", lambda data, **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(sys, "argv", ["dlm_lat_alt_mpi_run.py", "--output", str(tmp_path / "out.nc"), "--resume"])

    dlm_lat_alt_mpi_run.main()

    assert captured["resume"] is True
    assert captured["overwrite"] is False


def test_max_cells_fits_grid_in_batches(single_rank_mpi, monkeypatch, tmp_path):
    path = tmp_path / "grid.nc"
    fitted = []
    monkeypatch.setattr(
        dlmmc_mpi, "fit", lambda series, **kwargs: fitted.append(float(series.values[0])) or _posterior(1)
    )

    _fit_grid(path, resume=True, max_cells=2).close()
    assert fitted == [1.0, 2.0]
    with xr.open_dataset(path) as partial:
        np.testing.assert_array_equal(partial["fit_status"], [[1, 1, 0, 0]])

    estimate = estimate_grid_run(_data(), ranks=3, model="noregs_ar1", resume_from=path, max_cells=2)
    assert (estimate["fitted_cells"], estimate["completed_cells"], estimate["deferred_cells"]) == (1, 2, 0)

    result = _fit_grid(path, resume=True, max_cells=2)
    try:
        assert fitted == [1.0, 2.0, 4.0]
        np.testing.assert_array_equal(result["fit_status"], [[1, 1, 0, 1]])
        assert "resumed: 2 cells complete, 1 to fit" in result.attrs["history"]
    finally:
        result.close()


def test_estimate_reports_cells_deferred_by_max_cells():
    estimate = estimate_grid_run(_data(), ranks=3, model="noregs_ar1", max_cells=1)

    assert (estimate["fitted_cells"], estimate["deferred_cells"], estimate["skipped_cells"]) == (1, 2, 1)
    assert "Cells to fit: 1 of 4 (2 left for later runs, 1 skipped" in format_grid_estimate(estimate)
    with pytest.raises(ValueError, match="max_cells must be at least 1"):
        estimate_grid_run(_data(), ranks=3, model="noregs_ar1", max_cells=0)


def test_history_notes_cells_left_by_max_cells(single_rank_mpi, monkeypatch, tmp_path):
    path = tmp_path / "grid.nc"
    monkeypatch.setattr(dlmmc_mpi, "fit", lambda series, **kwargs: _posterior(1))
    _fit_grid(path, max_cells=1).close()
    result = _fit_grid(path, resume=True, max_cells=1)
    try:
        assert "resumed: 1 cells complete, 1 to fit (1 left for later runs)" in result.attrs["history"]
    finally:
        result.close()


def test_runner_forwards_max_cells_and_rejects_cell_indices(monkeypatch, tmp_path):
    from pathlib import Path

    import dlm_lat_alt_mpi_run

    captured = {}
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(
        dlm_lat_alt_mpi_run.dlmmc, "fit_grid_mpi", lambda data, **kwargs: captured.update(kwargs)
    )
    argv = ["dlm_lat_alt_mpi_run.py", "--output", str(tmp_path / "out.nc"), "--resume"]
    monkeypatch.setattr(sys, "argv", [*argv, "--max-cells", "50"])
    dlm_lat_alt_mpi_run.main()
    assert captured["max_cells"] == 50

    for extra in (["--max-cells", "0"], ["--max-cells", "5", "--cell-indices", "0"]):
        monkeypatch.setattr(sys, "argv", [*argv, *extra])
        with pytest.raises(SystemExit):
            dlm_lat_alt_mpi_run.main()
