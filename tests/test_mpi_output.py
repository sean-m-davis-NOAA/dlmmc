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


@pytest.mark.parametrize(
    "output_variables",
    [None, {"trend": "single", "sigma_AR": "double"}, {}],
)
def test_output_selection_precision_and_status(single_rank_mpi, monkeypatch, tmp_path, output_variables):
    import netCDF4

    time_values = np.arange(36, dtype=np.float64).reshape(2, 3, 6) / 7
    scalar_values = np.full((2, 3), np.pi)
    posterior = xr.Dataset({
        **{
            name: (("chain", "draw", "time"), time_values)
            for name in ("trend", "slope", "ar", "seasonal")
        },
        **{
            name: (("chain", "draw"), scalar_values)
            for name in ("sigma_trend", "sigma_AR", "rhoAR1", "sigma_seas")
        },
    })

    def fake_fit(series, **kwargs):
        assert "output_variables" not in kwargs
        if series.values[0] == 2:
            raise RuntimeError("sampler failed")
        return posterior

    monkeypatch.setattr(dlmmc_mpi, "fit", fake_fit)
    values = np.ones((6, 1, 3))
    values[:, 0, 1] = np.nan
    values[:, 0, 2] = 2
    data = xr.DataArray(
        values,
        dims=("time", "pressure", "latitude"),
        coords={
            "time": np.arange("2000-01-01", "2000-01-07", dtype="datetime64[D]"),
            "pressure": np.array([10], dtype=np.float64),
            "latitude": [-30, 0, 30],
        },
    )
    path = tmp_path / "grid.nc"
    result = dlmmc_mpi.fit_grid_mpi(
        data, path, model="noregs_ar1", chains=2, samples=3, output_variables=output_variables,
        posterior_means=False,
    )
    expected = output_variables if output_variables is not None else {
        name: "single" for name in posterior.data_vars
    }
    try:
        assert set(result.data_vars) == {*expected, "fit_status", "divergent_transitions"}
        np.testing.assert_array_equal(result["fit_status"], [[1, 0, -1]])
        for name, precision in expected.items():
            dtype = np.dtype("float32" if precision == "single" else "float64")
            assert result[name].dtype == dtype
            np.testing.assert_array_equal(
                result[name].isel(pressure=0, latitude=0), posterior[name].values.astype(dtype)
            )
            assert np.isnan(result[name].isel(pressure=0, latitude=1)).all()
            assert np.isnan(result[name].isel(pressure=0, latitude=2)).all()
        np.testing.assert_array_equal(result.time.values, data.time.values)
    finally:
        result.close()
    with netCDF4.Dataset(path) as output:
        assert output["time"].dtype == np.dtype("float64")
        assert output["pressure"].dtype == np.dtype("float64")
        assert output["fit_status"].dtype == np.dtype("int8")
        for name, precision in expected.items():
            assert output[name].dtype == np.dtype("float32" if precision == "single" else "float64")
    assert posterior["trend"].dtype == np.dtype("float64")


@pytest.mark.parametrize(
    ("output_variables", "message"),
    [
        ({"typo": "single"}, "not available"),
        ({"beta": "single"}, "not available"),
        ({"time": "single"}, "not available"),
        ({"fit_status": "single"}, "not available"),
        ({"trend": "half"}, "must be 'single' or 'double'"),
        (["trend"], "must map posterior variable names"),
    ],
)
def test_invalid_output_config_preserves_existing_file(
    single_rank_mpi, monkeypatch, tmp_path, output_variables, message
):
    def unexpected_compile(*args, **kwargs):
        pytest.fail("invalid output configuration must fail before model compilation")

    monkeypatch.setattr(dlmmc_mpi, "_get_compiled_model", unexpected_compile)
    path = tmp_path / "existing.nc"
    path.write_bytes(b"existing output")
    data = xr.DataArray(np.ones((6, 1, 1)), dims=("time", "pressure", "latitude"))
    with pytest.raises(RuntimeError, match=message):
        dlmmc_mpi.fit_grid_mpi(
            data, path, model="noregs_ar1", output_variables=output_variables, overwrite=True
        )
    assert path.read_bytes() == b"existing output"


@pytest.mark.parametrize(
    ("model", "name", "dims", "shape"),
    [
        ("vanilla_ar2", "rhoAR2", ("chain", "draw"), (2, 3)),
        ("vanilla_ar1", "beta", ("chain", "draw", "regressor"), (2, 3, 1)),
        ("dynregs_ar1", "beta", ("chain", "draw", "time", "regressor"), (2, 3, 6, 1)),
        ("dynregs_ar1", "sigma_reg", ("chain", "draw", "regressor"), (2, 3, 1)),
    ],
)
def test_model_specific_output_dimensions(single_rank_mpi, monkeypatch, tmp_path, model, name, dims, shape):
    posterior = xr.Dataset({name: (dims, np.full(shape, np.pi))})
    monkeypatch.setattr(dlmmc_mpi, "fit", lambda *args, **kwargs: posterior)
    data = xr.DataArray(np.ones((6, 1, 1)), dims=("time", "pressure", "latitude"))
    result = dlmmc_mpi.fit_grid_mpi(
        data,
        tmp_path / "grid.nc",
        model=model,
        regressors=np.ones((6, 1)),
        chains=2,
        samples=3,
        output_variables={name: "single"},
        posterior_means=False,
    )
    try:
        assert result[name].dims == dims + ("pressure", "latitude")
        assert result[name].dtype == np.dtype("float32")
        np.testing.assert_array_equal(
            result[name].isel(pressure=0, latitude=0), posterior[name].values.astype("float32")
        )
        assert result["fit_status"].item() == 1
    finally:
        result.close()


def test_worker_filters_and_casts_payload_using_writer_schema(single_rank_mpi, monkeypatch, tmp_path):
    from mpi4py import MPI

    schema = {"trend": ("f4", ("chain", "draw", "time", "pressure", "latitude"))}
    sent = []

    class WorkerComm(_SingleRankComm):
        def __init__(self):
            self.broadcasts = iter([None, schema, np.array([0])])

        def Get_rank(self):
            return 1

        def Get_size(self):
            return 2

        def bcast(self, value, root=0):
            return next(self.broadcasts)

        def send(self, value, dest, tag):
            sent.append((value, dest, tag))

    monkeypatch.setattr(MPI, "COMM_WORLD", WorkerComm())
    posterior = xr.Dataset({
        "trend": (("chain", "draw", "time"), np.full((1, 2, 6), np.pi)),
        "sigma_AR": (("chain", "draw"), np.ones((1, 2))),
    })
    monkeypatch.setattr(dlmmc_mpi, "fit", lambda *args, **kwargs: posterior)
    data = xr.DataArray(np.ones((6, 1, 1)), dims=("time", "pressure", "latitude"))
    result = dlmmc_mpi.fit_grid_mpi(data, tmp_path / "grid.nc", model="noregs_ar1", posterior_means=False)
    assert result is None
    (index, payload, error), dest, tag = sent[0]
    assert index == (0, 0)
    assert error is None
    assert (dest, tag) == (0, 1)
    assert set(payload) == {"trend"}
    assert payload["trend"].dtype == np.dtype("float32")


def test_single_precision_reduces_file_size_for_identical_draws(single_rank_mpi, monkeypatch, tmp_path):
    draws = np.random.default_rng(42).normal(size=(2, 32, 128))
    posterior = xr.Dataset({"trend": (("chain", "draw", "time"), draws)})
    monkeypatch.setattr(dlmmc_mpi, "fit", lambda *args, **kwargs: posterior)
    data = xr.DataArray(np.ones((128, 1, 1)), dims=("time", "pressure", "latitude"))
    sizes = {}
    for precision in ("single", "double"):
        path = tmp_path / f"{precision}.nc"
        result = dlmmc_mpi.fit_grid_mpi(
            data,
            path,
            model="noregs_ar1",
            chains=2,
            samples=32,
            output_variables={"trend": precision},
            posterior_means=False,
        )
        result.close()
        sizes[precision] = path.stat().st_size
    assert sizes["single"] < sizes["double"]


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_cell_chunks_bound_size_without_crossing_grid_cells(dtype):
    sizes = dict(chain=4, draw=10000, time=10000, regressor=5, pressure=27, latitude=12)
    dims = tuple(sizes)
    chunks = dlmmc_mpi._cell_chunks(dims, sizes, ("pressure", "latitude"), dtype)

    assert chunks[-2:] == (1, 1)
    assert all(1 <= chunk <= sizes[dim] for dim, chunk in zip(dims, chunks))
    assert np.prod(chunks) * np.dtype(dtype).itemsize <= 4 * 1024**2


def test_cell_writes_do_not_bloat_compressed_file_on_resume(single_rank_mpi, monkeypatch, tmp_path):
    import netCDF4

    names = ("trend", "slope", "ar", "seasonal")
    draws = np.random.default_rng(42).normal(size=(2, 64, 96, 4, 4)).astype("f4")

    def fake_fit(series, **kwargs):
        i, j = divmod(int(series.values[0]), 4)
        return xr.Dataset({
            name: (("chain", "draw", "time"), draws[:, :, :, i, j])
            for name in names
        })

    monkeypatch.setattr(dlmmc_mpi, "fit", fake_fit)
    data = xr.DataArray(
        np.broadcast_to(np.arange(16).reshape(1, 4, 4), (96, 4, 4)),
        dims=("time", "pressure", "latitude"),
    )
    path = tmp_path / "grid.nc"
    options = dict(
        model="noregs_ar1", chains=2, samples=64,
        output_variables={name: "single" for name in names},
        posterior_means=False, max_cells=8,
    )
    dlmmc_mpi.fit_grid_mpi(data, path, **options).close()
    dlmmc_mpi.fit_grid_mpi(data, path, resume=True, **options).close()

    assert path.stat().st_size < draws.nbytes * len(names) * 1.25
    with netCDF4.Dataset(path) as output:
        np.testing.assert_array_equal(output["fit_status"][:], np.ones((4, 4)))
        for name in names:
            assert output[name].chunking()[-2:] == [1, 1]
            np.testing.assert_array_equal(output[name][:], draws)


def test_runner_forwards_output_variables(monkeypatch, tmp_path):
    from pathlib import Path

    import dlm_lat_alt_mpi_run

    captured = {}

    def fake_fit_grid_mpi(data, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(dlm_lat_alt_mpi_run.dlmmc, "fit_grid_mpi", fake_fit_grid_mpi)
    monkeypatch.setattr(sys, "argv", ["dlm_lat_alt_mpi_run.py", "--output", str(tmp_path / "out.nc")])

    dlm_lat_alt_mpi_run.main()

    assert captured["output_variables"] is dlm_lat_alt_mpi_run.OUTPUT_VARIABLES


def test_grid_records_divergences_and_run_metadata(single_rank_mpi, monkeypatch, tmp_path):
    import netCDF4

    def fake_fit(series, **kwargs):
        if series.values[0] == 2:
            raise RuntimeError("sampler failed")
        posterior = xr.Dataset({"sigma_AR": (("chain", "draw"), np.ones((2, 3)))})
        posterior.attrs["divergent_transitions"] = int(series.values[0] * 10)
        return posterior

    monkeypatch.setattr(dlmmc_mpi, "fit", fake_fit)
    values = np.ones((6, 1, 4))
    values[:, 0, 1] = np.nan
    values[:, 0, 2] = 2
    values[:, 0, 3] = 3
    data = xr.DataArray(values, dims=("time", "pressure", "latitude"))
    path = tmp_path / "grid.nc"
    result = dlmmc_mpi.fit_grid_mpi(
        data,
        path,
        model="dynregs_ar1",
        regressors=np.ones((6, 2)),
        chains=2,
        warmup=4,
        samples=3,
        seed=11,
        sigma_reg_prior=[0.1, 0.2],
        adapt_delta=0.9,
        output_variables={"sigma_AR": "single"},
        posterior_means=False,
    )
    try:
        np.testing.assert_array_equal(result["fit_status"], [[1, 0, -1, 1]])
        np.testing.assert_array_equal(result["divergent_transitions"], [[10, np.nan, np.nan, 30]])
        attrs = result.attrs
        assert attrs["model"] == "dynregs_ar1"
        assert (attrs["chains"], attrs["warmup"], attrs["samples_per_chain"], attrs["seed"]) == (2, 4, 3, 11)
        assert attrs["adapt_delta"] == 0.9
        np.testing.assert_allclose(attrs["prior_sigma_reg"], [0.1, 0.2])
        assert attrs["prior_sigma_AR"] == 0.5
        assert "writer" in attrs
        assert "divergent" in result["divergent_transitions"].attrs["comment"]
    finally:
        result.close()
    with netCDF4.Dataset(path) as output:
        assert output["divergent_transitions"].dtype == np.dtype("int32")


def test_invalid_prior_fails_before_compilation(single_rank_mpi, monkeypatch, tmp_path):
    def unexpected_compile(*args, **kwargs):
        pytest.fail("invalid priors must fail before model compilation")

    monkeypatch.setattr(dlmmc_mpi, "_get_compiled_model", unexpected_compile)
    data = xr.DataArray(np.ones((6, 1, 1)), dims=("time", "pressure", "latitude"))
    with pytest.raises(RuntimeError, match="sigma_AR_prior"):
        dlmmc_mpi.fit_grid_mpi(data, tmp_path / "grid.nc", model="noregs_ar1", sigma_AR_prior=-1.0)


def test_posterior_means_cover_non_time_parameters_even_when_draws_are_omitted(
    single_rank_mpi, monkeypatch, tmp_path
):
    import netCDF4

    rng = np.random.default_rng(3)
    posterior = xr.Dataset({
        **{name: (("chain", "draw", "time"), rng.normal(size=(2, 3, 6)))
           for name in ("trend", "slope", "ar", "seasonal")},
        **{name: (("chain", "draw"), rng.normal(size=(2, 3)))
           for name in ("sigma_trend", "sigma_AR", "rhoAR1", "sigma_seas")},
        "beta": (("chain", "draw", "regressor"), rng.normal(size=(2, 3, 2))),
    })

    def fake_fit(series, **kwargs):
        if series.values[0] == 2:
            raise RuntimeError("sampler failed")
        return posterior

    monkeypatch.setattr(dlmmc_mpi, "fit", fake_fit)
    values = np.ones((6, 1, 3))
    values[:, 0, 1] = np.nan
    values[:, 0, 2] = 2
    data = xr.DataArray(values, dims=("time", "pressure", "latitude"))
    path = tmp_path / "grid.nc"
    result = dlmmc_mpi.fit_grid_mpi(
        data, path, model="vanilla_ar1", regressors=np.ones((6, 2)), chains=2, samples=3,
        output_variables={"trend": "single"},
    )
    try:
        means = {name for name in result.data_vars if name.endswith("_mean")}
        assert means == {"sigma_trend_mean", "sigma_AR_mean", "rhoAR1_mean", "sigma_seas_mean", "beta_mean"}
        assert "sigma_AR" not in result
        assert result["sigma_AR_mean"].dims == ("pressure", "latitude")
        assert result["beta_mean"].dims == ("regressor", "pressure", "latitude")
        np.testing.assert_allclose(
            result["sigma_AR_mean"].isel(pressure=0, latitude=0), posterior["sigma_AR"].mean()
        )
        np.testing.assert_allclose(
            result["beta_mean"].isel(pressure=0, latitude=0), posterior["beta"].mean(("chain", "draw"))
        )
        assert np.isnan(result["sigma_AR_mean"].isel(pressure=0, latitude=[1, 2])).all()
        assert "posterior mean" in result["sigma_AR_mean"].attrs["long_name"]
    finally:
        result.close()
    with netCDF4.Dataset(path) as output:
        assert output["sigma_AR_mean"].dtype == np.dtype("float64")
        for name in means:
            assert output[name].chunking()[-2:] == [1, 1]


def test_posterior_means_can_be_disabled(single_rank_mpi, monkeypatch, tmp_path):
    posterior = xr.Dataset({"sigma_AR": (("chain", "draw"), np.ones((1, 2)))})
    monkeypatch.setattr(dlmmc_mpi, "fit", lambda *args, **kwargs: posterior)
    data = xr.DataArray(np.ones((6, 1, 1)), dims=("time", "pressure", "latitude"))
    result = dlmmc_mpi.fit_grid_mpi(
        data, tmp_path / "grid.nc", model="noregs_ar1", chains=1, samples=2,
        output_variables={"sigma_AR": "single"}, posterior_means=False,
    )
    try:
        assert not [name for name in result.data_vars if name.endswith("_mean")]
        assert result["fit_status"].item() == 1
    finally:
        result.close()
