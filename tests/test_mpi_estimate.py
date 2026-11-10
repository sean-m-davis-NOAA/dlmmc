import sys

import numpy as np
import pytest
import xarray as xr

from dlmmc.mpi import estimate_grid_run, format_grid_estimate, parse_time_limit


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("90", 5400),
        ("90:30", 5430),
        ("4:00:00", 14400),
        ("1-2", 93600),
        ("1-02:30", 95400),
        ("1-02:30:15", 95415),
    ],
)
def test_parse_time_limit_accepts_slurm_formats(value, seconds):
    assert parse_time_limit(value) == seconds


@pytest.mark.parametrize("value", ["", "1:2:3:4", "abc", "1-", "-1"])
def test_parse_time_limit_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_time_limit(value)


def _grid():
    values = np.random.default_rng(0).normal(size=(4, 2, 5))
    values[:, 0, 1] = np.nan
    values[:, 1, 3] = 0.0
    return xr.DataArray(values, dims=("time", "pressure", "latitude"))


def test_estimate_reproduces_static_assignment_and_flags_constant_cells():
    estimate = estimate_grid_run(_grid(), ranks=4, model="noregs_ar1", chains=2, samples=10)

    assert estimate["grid_cells"] == 10
    assert estimate["fitted_cells"] == 9
    assert estimate["skipped_cells"] == 1
    assert estimate["constant_cell_ids"] == [8]
    # Flat IDs 0,2..9 over three workers: residues 0 -> {0,3,6,9}, 1 -> {4,7}, 2 -> {2,5,8}.
    assert (estimate["fitting_workers"], estimate["max_cells_per_worker"], estimate["min_cells_per_worker"]) == (3, 4, 2)
    assert estimate["estimated_seconds"] is None


def test_single_rank_fits_every_cell_itself():
    estimate = estimate_grid_run(_grid(), ranks=1, model="noregs_ar1", seconds_per_cell=10, safety_factor=1)

    assert estimate["fitting_workers"] == 1
    assert estimate["max_cells_per_worker"] == 9
    assert estimate["estimated_seconds"] == 90


def test_idle_workers_are_reported():
    data = xr.DataArray(np.random.default_rng(0).normal(size=(4, 2, 1)), dims=("time", "pressure", "latitude"))
    estimate = estimate_grid_run(data, ranks=4, model="noregs_ar1")

    assert estimate["fitted_cells"] == 2
    assert estimate["idle_workers"] == 1
    assert estimate["min_cells_per_worker"] == 0


def test_time_estimate_and_limit_check():
    kwargs = dict(ranks=4, model="noregs_ar1", seconds_per_cell=100, safety_factor=1.5)
    ok = estimate_grid_run(_grid(), time_limit="0:10:00", **kwargs)
    short = estimate_grid_run(_grid(), time_limit=500, **kwargs)

    assert ok["estimated_seconds"] == 400
    assert ok["estimated_seconds_with_margin"] == 600
    assert ok["fits_time_limit"] is True
    assert short["fits_time_limit"] is False
    assert "WARNING: time limit" in format_grid_estimate(short)
    assert "OK" in format_grid_estimate(ok)


def test_output_size_counts_selected_precision_and_means():
    data = _grid()
    kwargs = dict(ranks=2, model="noregs_ar1", chains=2, samples=10)
    single = estimate_grid_run(data, output_variables={"sigma_AR": "single"}, posterior_means=False, **kwargs)
    double = estimate_grid_run(data, output_variables={"sigma_AR": "double"}, posterior_means=False, **kwargs)
    trend = estimate_grid_run(data, output_variables={"trend": "single"}, posterior_means=False, **kwargs)
    means = estimate_grid_run(data, output_variables={}, **kwargs)

    assert single["uncompressed_output_bytes"] == 4 * 2 * 10 * 10
    assert double["uncompressed_output_bytes"] == 2 * single["uncompressed_output_bytes"]
    assert trend["uncompressed_output_bytes"] == 4 * 2 * 10 * 4 * 10
    # noregs_ar1 has four non-time parameters: sigma_trend, sigma_AR, rhoAR1, sigma_seas.
    assert means["uncompressed_output_bytes"] == 8 * 4 * 10


def test_regressor_dimension_is_sized():
    estimate = estimate_grid_run(
        _grid(), ranks=2, model="vanilla_ar1", regressors=np.ones((4, 3)), chains=1, samples=2,
        output_variables={"beta": "single"}, posterior_means=False,
    )
    assert estimate["uncompressed_output_bytes"] == 4 * 1 * 2 * 3 * 10


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ranks": 0}, "ranks"),
        ({"ranks": 2, "seconds_per_cell": 0}, "seconds_per_cell"),
        ({"ranks": 2, "safety_factor": 0.5}, "safety_factor"),
        ({"ranks": 2, "output_variables": {"typo": "single"}}, "not available"),
    ],
)
def test_invalid_estimate_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        estimate_grid_run(_grid(), model="noregs_ar1", **kwargs)


def test_runner_dry_run_does_not_fit(monkeypatch, tmp_path, capsys):
    from pathlib import Path

    import dlm_lat_alt_mpi_run

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(
        dlm_lat_alt_mpi_run.dlmmc, "fit_grid_mpi", lambda *a, **k: pytest.fail("dry run must not fit")
    )
    monkeypatch.setattr(
        sys, "argv",
        ["dlm_lat_alt_mpi_run.py", "--dry-run", "--ranks", "20", "--seconds-per-cell", "568",
         "--time-limit", "4:00:00", "--output", str(tmp_path / "out.nc")],
    )

    dlm_lat_alt_mpi_run.main()

    report = capsys.readouterr().out
    assert "Cells to fit: 324 of 324" in report
    assert "19 fitting workers" in report
    assert "Time limit 4:00:00: OK." in report
    assert not (tmp_path / "out.nc").exists()


def test_runner_dry_run_requires_ranks(monkeypatch):
    import dlm_lat_alt_mpi_run

    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.setattr(sys, "argv", ["dlm_lat_alt_mpi_run.py", "--dry-run"])
    with pytest.raises(SystemExit):
        dlm_lat_alt_mpi_run.main()
