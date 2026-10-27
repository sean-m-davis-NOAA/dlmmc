"""MPI grid fitting with a single NetCDF writer on rank zero."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .api import _MODEL_SPECS, _get_compiled_model, _prepare_regressors, _resolve_model, fit


def _regressor_count(regressors: Any) -> int:
    if regressors is None:
        return 0
    if isinstance(regressors, xr.Dataset):
        return len(regressors.data_vars)
    if isinstance(regressors, xr.DataArray):
        return regressors.sizes.get("regressor", 1)
    values = np.asarray(regressors)
    return 1 if values.ndim == 1 else values.shape[1]


def _coordinate_values(data: xr.DataArray, dim: str) -> np.ndarray:
    if dim in data.coords:
        return data.coords[dim].to_numpy()
    return np.arange(data.sizes[dim])


def _create_result_file(
    path: Path,
    data: xr.DataArray,
    grid_dims: tuple[str, ...],
    nreg: int,
    regressor_names: list[str],
    model_name: str,
    spec: Mapping[str, Any],
    chains: int,
    samples: int,
) -> None:
    try:
        import netCDF4
    except ImportError as exc:
        raise ImportError("fit_grid_mpi requires netCDF4; install dlmmc[mpi].") from exc

    with netCDF4.Dataset(path, "w", format="NETCDF4") as output:
        output.createDimension("chain", chains)
        output.createDimension("draw", samples)
        output.createDimension("time", data.sizes["time"])
        for dim in grid_dims:
            output.createDimension(dim, data.sizes[dim])
        if nreg:
            output.createDimension("regressor", nreg)

        for name, values in (
            ("chain", np.arange(1, chains + 1)),
            ("draw", np.arange(1, samples + 1)),
            ("time", data.coords["time"].to_numpy() if "time" in data.coords else np.arange(data.sizes["time"])),
        ):
            values = np.asarray(values)
            if np.issubdtype(values.dtype, np.datetime64):
                coordinate = output.createVariable(name, "f8", (name,))
                coordinate.units = "seconds since 1970-01-01 00:00:00"
                coordinate.calendar = "proleptic_gregorian"
                coordinate[:] = netCDF4.date2num(
                    values.astype("datetime64[us]").astype(object).tolist(),
                    coordinate.units,
                    calendar=coordinate.calendar,
                )
            else:
                coordinate = output.createVariable(name, values.dtype, (name,))
                coordinate[:] = values
        for dim in grid_dims:
            values = np.asarray(_coordinate_values(data, dim))
            if values.dtype.kind in {"U", "O"}:
                coordinate = output.createVariable(dim, str, (dim,))
                coordinate[:] = values.astype(str)
            else:
                coordinate = output.createVariable(dim, values.dtype, (dim,))
                coordinate[:] = values
        if nreg:
            regressor_coordinate = output.createVariable("regressor", str, ("regressor",))
            regressor_coordinate[:] = np.asarray(regressor_names, dtype=str)

        grid_suffix = grid_dims
        scalar_dims = ("chain", "draw") + grid_suffix
        time_dims = ("chain", "draw", "time") + grid_suffix
        variables: dict[str, tuple[str, ...]] = {
            "trend": time_dims,
            "slope": time_dims,
            "ar": time_dims,
            "sigma_trend": scalar_dims,
            "sigma_AR": scalar_dims,
            "rhoAR1": scalar_dims,
        }
        if spec["seasonal"]:
            variables["seasonal"] = time_dims
            variables["sigma_seas"] = scalar_dims
        if spec["ar2"]:
            variables["rhoAR2"] = scalar_dims
        if spec["regressors"] != "none":
            if spec["regressors"] == "dynamic":
                variables["beta"] = ("chain", "draw", "time", "regressor") + grid_suffix
                variables["sigma_reg"] = ("chain", "draw", "regressor") + grid_suffix
            else:
                variables["beta"] = ("chain", "draw", "regressor") + grid_suffix

        for name, dims in variables.items():
            output.createVariable(name, "f8", dims, zlib=True, fill_value=np.nan)
        status = output.createVariable("fit_status", "i1", grid_dims)
        status[:] = 0
        status.flag_values = np.array([-1, 0, 1], dtype=np.int8)
        status.flag_meanings = "failed skipped complete"
        output.model = model_name
        output.backend = "cmdstanpy"
        output.writer = "MPI rank 0; worker fits streamed as they complete"


def _write_result(output: Any, index: tuple[int, ...], posterior: dict[str, np.ndarray] | None, error: str | None) -> None:
    if error is not None:
        output["fit_status"][index] = -1
        print(f"Fit failed at grid index {index}: {error}")
        return
    if posterior is None:
        output["fit_status"][index] = 0
        return

    for name, values in posterior.items():
        dims = output[name].dimensions
        extra_dims = len(dims) - len(index)
        selection = (slice(None),) * extra_dims + index
        output[name][selection] = values
    output["fit_status"][index] = 1


def fit_grid_mpi(
    data: xr.DataArray,
    output: str | os.PathLike[str],
    *,
    uncertainty: xr.DataArray | None = None,
    regressors: xr.DataArray | xr.Dataset | np.ndarray | None = None,
    model: str = "vanilla_ar1",
    grid_dims: Sequence[str] | None = None,
    overwrite: bool = False,
    **fit_options: Any,
) -> xr.Dataset | None:
    """Fit a time series at each grid cell and stream results to one NetCDF file.

    Call this function on every MPI rank. Rank 0 reserves itself as the sole
    NetCDF writer; the other ranks perform fits and send results as they finish.
    Rank 0 returns the completed xarray Dataset; worker ranks return ``None``.
    With a single rank, rank 0 performs the fits serially.

    ``data`` must have dimensions ``time`` plus the grid dimensions to fit over.
    ``uncertainty`` and xarray ``regressors`` may be shared over time or include
    grid dimensions that are sliced for each fit. Existing output is protected
    unless ``overwrite=True``.
    """
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError("fit_grid_mpi requires mpi4py; install dlmmc[mpi].") from exc

    if not isinstance(data, xr.DataArray) or "time" not in data.dims:
        raise TypeError("data must be an xarray.DataArray with a 'time' dimension.")
    selected_grid_dims = tuple(grid_dims or (dim for dim in data.dims if dim != "time"))
    if not selected_grid_dims or set(data.dims) != {"time", *selected_grid_dims}:
        raise ValueError("grid_dims must name every non-time dimension of data.")
    data = data.transpose("time", *selected_grid_dims)
    for optional, name in ((uncertainty, "uncertainty"), (regressors, "regressors")):
        if isinstance(optional, (xr.DataArray, xr.Dataset)):
            unknown = set(optional.dims) - {"time", *selected_grid_dims}
            if unknown:
                raise ValueError(f"{name} contains dimensions outside data/grid_dims: {sorted(unknown)}")

    model_name, spec = _resolve_model(model)
    nreg = 0 if spec["regressors"] == "none" else _regressor_count(regressors)
    if spec["regressors"] != "none" and nreg == 0:
        raise ValueError(f"Model {model_name!r} requires regressors.")
    if spec["regressors"] == "none" and regressors is not None:
        raise ValueError(f"Model {model_name!r} does not accept regressors.")

    time = data.coords["time"] if "time" in data.coords else xr.DataArray(
        np.arange(data.sizes["time"]), dims=("time",), name="time"
    )
    if isinstance(regressors, xr.Dataset):
        regressor_names = list(regressors.data_vars)
    elif isinstance(regressors, xr.DataArray) and "regressor" in regressors.coords:
        regressor_names = [str(value) for value in regressors.coords["regressor"].values]
    else:
        regressor_names = [f"regressor_{index + 1}" for index in range(nreg)]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    process_count = comm.Get_size()
    output_path = Path(output).expanduser().resolve()
    cell_count = int(np.prod([data.sizes[dim] for dim in selected_grid_dims]))
    chains = int(fit_options.get("chains", 4))
    samples = int(fit_options.get("samples", 1000))
    setup_error = None

    if rank == 0:
        try:
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Output already exists: {output_path}; set overwrite=True to replace it.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                output_path.unlink()
            _get_compiled_model(model_name, spec, fit_options.get("cache_dir"))
            _create_result_file(
                output_path, data, selected_grid_dims, nreg, regressor_names,
                model_name, spec, chains, samples,
            )
        except Exception as exc:
            setup_error = str(exc)
    setup_error = comm.bcast(setup_error, root=0)
    if setup_error:
        raise RuntimeError(f"Could not initialize grid fit: {setup_error}")
    comm.Barrier()

    grid_shape = tuple(data.sizes[dim] for dim in selected_grid_dims)
    indices = np.arange(cell_count)
    if process_count == 1:
        local_indices = indices
    elif rank == 0:
        local_indices = np.empty(0, dtype=int)
    else:
        local_indices = indices[indices % (process_count - 1) == rank - 1]

    def run_cell(flat_index: int) -> tuple[tuple[int, ...], dict[str, np.ndarray] | None, str | None]:
        cell_index = tuple(int(value) for value in np.unravel_index(flat_index, grid_shape))
        indexers = dict(zip(selected_grid_dims, cell_index))
        series = data.isel(indexers)
        if not np.isfinite(series.to_numpy()).any():
            return cell_index, None, None
        cell_uncertainty = uncertainty
        if isinstance(uncertainty, xr.DataArray):
            cell_uncertainty = uncertainty.isel({dim: value for dim, value in indexers.items() if dim in uncertainty.dims})
        cell_regressors = regressors
        if isinstance(regressors, (xr.DataArray, xr.Dataset)):
            cell_regressors = regressors.isel({dim: value for dim, value in indexers.items() if dim in regressors.dims})
        try:
            posterior = fit(
                series,
                regressors=cell_regressors,
                model=model_name,
                uncertainty=cell_uncertainty,
                **fit_options,
            )
            return cell_index, {name: posterior[name].to_numpy() for name in posterior.data_vars}, None
        except Exception as exc:
            return cell_index, None, str(exc)

    if rank == 0:
        with __import__("netCDF4").Dataset(output_path, "r+") as result_file:
            if process_count == 1:
                for flat_index in local_indices:
                    cell_index, posterior_data, error = run_cell(int(flat_index))
                    _write_result(result_file, cell_index, posterior_data, error)
            else:
                for _ in range(cell_count):
                    cell_index, posterior_data, error = comm.recv(source=MPI.ANY_SOURCE, tag=1)
                    _write_result(result_file, cell_index, posterior_data, error)
    else:
        for flat_index in local_indices:
            comm.send(run_cell(int(flat_index)), dest=0, tag=1)

    comm.Barrier()
    if rank == 0:
        return xr.open_dataset(output_path)
    return None
