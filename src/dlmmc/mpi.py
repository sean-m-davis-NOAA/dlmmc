"""MPI grid fitting with a single NetCDF writer on rank zero."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from .api import (
    _MODEL_SPECS,
    _PRIOR_ARGUMENTS,
    DIVERGENCE_COMMENT,
    _get_compiled_model,
    _prepare_regressors,
    _resolve_model,
    _resolve_priors,
    _run_metadata,
    fit,
)


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


def _posterior_dims(grid_dims: tuple[str, ...], spec: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    scalar_dims = ("chain", "draw") + grid_dims
    time_dims = ("chain", "draw", "time") + grid_dims
    variables = {
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
            variables["beta"] = ("chain", "draw", "time", "regressor") + grid_dims
            variables["sigma_reg"] = ("chain", "draw", "regressor") + grid_dims
        else:
            variables["beta"] = ("chain", "draw", "regressor") + grid_dims
    return variables


def _mean_schema(variables: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Map each non-time-varying parameter to its per-cell posterior-mean variable."""
    return {
        f"{name}_mean": (name, tuple(dim for dim in dims if dim not in ("chain", "draw")))
        for name, dims in variables.items()
        if "time" not in dims
    }


def _output_schema(
    grid_dims: tuple[str, ...],
    spec: Mapping[str, Any],
    output_variables: Mapping[str, Literal["single", "double"]] | None,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    variables = _posterior_dims(grid_dims, spec)
    if output_variables is None:
        return {name: ("f4", dims) for name, dims in variables.items()}
    if not isinstance(output_variables, Mapping):
        raise TypeError("output_variables must map posterior variable names to 'single' or 'double'.")
    schema = {}
    for name, precision in output_variables.items():
        if name not in variables:
            raise ValueError(
                f"Output variable {name!r} is not available for this model. "
                f"Choose from: {', '.join(variables)}."
            )
        if precision not in ("single", "double"):
            raise ValueError(f"Precision for {name!r} must be 'single' or 'double', not {precision!r}.")
        schema[name] = ("f4" if precision == "single" else "f8", variables[name])
    return schema


def _create_result_file(
    path: Path,
    data: xr.DataArray,
    grid_dims: tuple[str, ...],
    nreg: int,
    regressor_names: list[str],
    metadata: Mapping[str, Any],
    output_schema: Mapping[str, tuple[str, tuple[str, ...]]],
    mean_schema: Mapping[str, tuple[str, tuple[str, ...]]],
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

        for name, (dtype, dims) in output_schema.items():
            output.createVariable(name, dtype, dims, zlib=True, fill_value=np.nan)
        for name, (source, dims) in mean_schema.items():
            mean = output.createVariable(name, "f8", dims, zlib=True, fill_value=np.nan)
            mean.long_name = f"posterior mean of {source}"
            mean.comment = (
                f"Mean of all post-warmup draws of {source} over every chain, computed from "
                "the full-precision draws before any output precision reduction. Missing "
                "for skipped or failed cells."
            )
        status = output.createVariable("fit_status", "i1", grid_dims)
        status[:] = 0
        status.flag_values = np.array([-1, 0, 1], dtype=np.int8)
        status.flag_meanings = "failed skipped complete"
        divergences = output.createVariable("divergent_transitions", "i4", grid_dims, fill_value=-1)
        divergences.long_name = "number of divergent post-warmup transitions"
        divergences.units = "1"
        divergences.comment = DIVERGENCE_COMMENT + " Missing for skipped or failed cells."
        output.setncatts(dict(metadata))
        output.writer = "MPI rank 0; worker fits streamed as they complete"


# Attributes that may legitimately differ between the original run and a resumed run.
_RESUME_IGNORED_ATTRIBUTES = {"date_created", "dlmmc_version", "cmdstan_version", "writer", "history"}


def _attribute_equal(left: Any, right: Any) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    return np.array_equal(np.asarray(left), np.asarray(right))


def _read_completed_cells(path: Path, grid_dims: tuple[str, ...]) -> np.ndarray:
    """Return a flat boolean mask of cells whose ``fit_status`` is complete."""
    with xr.open_dataset(path, decode_times=False) as existing:
        if "fit_status" not in existing:
            raise ValueError(f"{path} has no fit_status variable; it was not written by fit_grid_mpi.")
        status = existing["fit_status"]
        if set(status.dims) != set(grid_dims):
            raise ValueError(f"{path} fit_status has dimensions {status.dims}, expected {grid_dims}.")
        return (status.transpose(*grid_dims).to_numpy() == 1).ravel()


def _check_resumable(
    path: Path,
    data: xr.DataArray,
    grid_dims: tuple[str, ...],
    nreg: int,
    metadata: Mapping[str, Any],
    output_schema: Mapping[str, tuple[str, tuple[str, ...]]],
    mean_schema: Mapping[str, tuple[str, tuple[str, ...]]],
    chains: int,
    samples: int,
) -> None:
    """Raise ``ValueError`` unless ``path`` was written by an identically configured run."""
    import netCDF4

    problems = []
    with netCDF4.Dataset(path, "r") as existing:
        expected_sizes = {"chain": chains, "draw": samples, "time": data.sizes["time"]}
        expected_sizes.update({dim: data.sizes[dim] for dim in grid_dims})
        if nreg:
            expected_sizes["regressor"] = nreg
        for dim, size in expected_sizes.items():
            actual = len(existing.dimensions[dim]) if dim in existing.dimensions else None
            if actual != size:
                problems.append(f"dimension {dim!r} is {actual}, expected {size}")
        for dim in grid_dims:
            if dim in existing.variables and len(existing.variables[dim]) == data.sizes[dim]:
                stored = np.asarray(existing.variables[dim][:])
                if not np.array_equal(stored.astype(str), np.asarray(_coordinate_values(data, dim)).astype(str)):
                    problems.append(f"coordinate {dim!r} values differ")

        expected_variables = {name: np.dtype(dtype) for name, (dtype, _) in output_schema.items()}
        expected_variables.update({name: np.dtype("f8") for name in mean_schema})
        stored_variables = {
            name: variable.dtype
            for name, variable in existing.variables.items()
            if name not in existing.dimensions and name not in ("fit_status", "divergent_transitions")
        }
        if stored_variables != expected_variables:
            problems.append(
                "output variables differ: file has "
                + ", ".join(f"{name}:{dtype}" for name, dtype in sorted(stored_variables.items()))
                + "; run requests "
                + ", ".join(f"{name}:{dtype}" for name, dtype in sorted(expected_variables.items()))
            )

        stored_attributes = {
            name: existing.getncattr(name)
            for name in existing.ncattrs()
            if name not in _RESUME_IGNORED_ATTRIBUTES
        }
        for name in sorted(set(stored_attributes) | (set(metadata) - _RESUME_IGNORED_ATTRIBUTES)):
            if name not in stored_attributes or name not in metadata:
                problems.append(f"setting {name!r} is only set in the {'run' if name in metadata else 'file'}")
            elif not _attribute_equal(stored_attributes[name], metadata[name]):
                stored, requested = (
                    value.tolist() if isinstance(value, (np.ndarray, np.generic)) else value
                    for value in (stored_attributes[name], metadata[name])
                )
                problems.append(f"setting {name!r} is {stored!r} in the file but {requested!r} now")
    if problems:
        raise ValueError(
            f"Cannot resume {path}: it was written with different settings ("
            + "; ".join(problems)
            + "). Use the original settings, or set overwrite=True to start over."
        )


def _write_result_to_file(
    path: Path, index: tuple[int, ...], posterior: dict[str, np.ndarray] | None, error: str | None
) -> None:
    # Open the file for each cell so an interrupted run (for example at a wall-clock
    # limit) leaves a closed, readable file that can be resumed.
    import netCDF4

    with netCDF4.Dataset(path, "r+") as output:
        _write_result(output, index, posterior, error)


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


def parse_time_limit(value: str) -> int:
    """Convert a Slurm ``--time`` value to seconds.

    Accepts ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM``, and
    ``D-HH:MM:SS``.
    """
    text = str(value).strip()
    try:
        if "-" in text:
            days_text, rest = text.split("-", 1)
            days = int(days_text)
            parts = [int(part) for part in rest.split(":")]
            if not 1 <= len(parts) <= 3:
                raise ValueError
            hours, minutes, seconds = (parts + [0, 0])[:3]
        else:
            days = 0
            parts = [int(part) for part in text.split(":")]
            if len(parts) == 1:
                hours, minutes, seconds = 0, parts[0], 0
            elif len(parts) == 2:
                hours, minutes, seconds = 0, parts[0], parts[1]
            elif len(parts) == 3:
                hours, minutes, seconds = parts
            else:
                raise ValueError
    except ValueError:
        raise ValueError(f"Unrecognized Slurm time limit {value!r}.") from None
    if min(days, hours, minutes, seconds) < 0:
        raise ValueError(f"Time limit must not be negative: {value!r}.")
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def estimate_grid_run(
    data: xr.DataArray,
    ranks: int,
    *,
    regressors: xr.DataArray | xr.Dataset | np.ndarray | None = None,
    model: str = "vanilla_ar1",
    grid_dims: Sequence[str] | None = None,
    output_variables: Mapping[str, Literal["single", "double"]] | None = None,
    posterior_means: bool = True,
    chains: int = 4,
    samples: int = 1000,
    seconds_per_cell: float | None = None,
    safety_factor: float = 1.25,
    time_limit: str | int | None = None,
    resume_from: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Preview a ``fit_grid_mpi`` run without fitting or initializing MPI.

    Counts the cells that would be fitted (cells with at least one finite value)
    and lists those whose finite values are all equal (``constant_cell_ids``),
    reproduces the round-robin assignment of cells to ``ranks - 1`` workers (or
    to rank 0 when ``ranks == 1``), and estimates the uncompressed output size.
    With ``resume_from`` naming an existing output file, cells already complete
    there are excluded (``completed_cells``), as in ``fit_grid_mpi(resume=True)``;
    a missing file means a fresh run.

    The busiest worker determines the run time, so with ``seconds_per_cell`` the
    estimate is ``max cells per worker * seconds_per_cell * safety_factor``.
    ``seconds_per_cell`` is not predicted: measure it for the same model,
    chains, iterations, ``parallel_chains``, and node layout (for example, from a
    previous run's wall time divided by its busiest worker's cell count).
    ``time_limit`` (seconds or a Slurm ``--time`` string) is compared with the
    estimate, and ``fits_time_limit`` is ``False`` when it is too short.
    """
    if ranks < 1:
        raise ValueError("ranks must be at least 1.")
    if seconds_per_cell is not None and seconds_per_cell <= 0:
        raise ValueError("seconds_per_cell must be positive.")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least 1.")
    if not isinstance(data, xr.DataArray) or "time" not in data.dims:
        raise TypeError("data must be an xarray.DataArray with a 'time' dimension.")
    selected_grid_dims = tuple(grid_dims or (dim for dim in data.dims if dim != "time"))
    if not selected_grid_dims or set(data.dims) != {"time", *selected_grid_dims}:
        raise ValueError("grid_dims must name every non-time dimension of data.")
    data = data.transpose("time", *selected_grid_dims)
    _, spec = _resolve_model(model)
    nreg = 0 if spec["regressors"] == "none" else _regressor_count(regressors)

    values = data.to_numpy().reshape(data.sizes["time"], -1)
    has_data = np.isfinite(values).any(axis=0)
    completed = np.zeros(has_data.size, dtype=bool)
    if resume_from is not None and Path(resume_from).expanduser().exists():
        completed = _read_completed_cells(Path(resume_from).expanduser(), selected_grid_dims)
        if completed.size != has_data.size:
            raise ValueError(f"{resume_from} has {completed.size} cells but data has {has_data.size}.")
    fitted = has_data & ~completed
    flat_ids = np.flatnonzero(fitted)
    with np.errstate(invalid="ignore"):
        constant = fitted & (np.nanmax(np.where(np.isfinite(values), values, -np.inf), axis=0)
                             == np.nanmin(np.where(np.isfinite(values), values, np.inf), axis=0))
    workers = 1 if ranks == 1 else ranks - 1
    per_worker = np.bincount(np.arange(flat_ids.size) % workers, minlength=workers)

    sizes = {**{dim: data.sizes[dim] for dim in data.dims}, "chain": chains, "draw": samples, "regressor": nreg}
    variables = _posterior_dims(selected_grid_dims, spec)
    schema = _output_schema(selected_grid_dims, spec, output_variables)
    means = _mean_schema(variables) if posterior_means else {}
    output_bytes = sum(
        np.dtype(dtype).itemsize * int(np.prod([sizes[dim] for dim in dims]))
        for dtype, dims in schema.values()
    ) + sum(8 * int(np.prod([sizes[dim] for dim in dims])) for _, dims in means.values())

    estimate: dict[str, Any] = {
        "grid_cells": int(fitted.size),
        "fitted_cells": int(flat_ids.size),
        "completed_cells": int(completed.sum()),
        "skipped_cells": int(np.sum(~has_data & ~completed)),
        "ranks": ranks,
        "fitting_workers": workers,
        "constant_cell_ids": np.flatnonzero(constant).tolist(),
        "max_cells_per_worker": int(per_worker.max()) if per_worker.size else 0,
        "min_cells_per_worker": int(per_worker.min()) if per_worker.size else 0,
        "idle_workers": int(np.sum(per_worker == 0)),
        "uncompressed_output_bytes": int(output_bytes),
        "estimated_seconds": None,
        "estimated_seconds_with_margin": None,
        "time_limit_seconds": None,
        "fits_time_limit": None,
    }
    if seconds_per_cell is not None:
        seconds = estimate["max_cells_per_worker"] * float(seconds_per_cell)
        estimate["estimated_seconds"] = seconds
        estimate["estimated_seconds_with_margin"] = seconds * safety_factor
    if time_limit is not None:
        limit = time_limit if isinstance(time_limit, int) else parse_time_limit(time_limit)
        estimate["time_limit_seconds"] = limit
        if estimate["estimated_seconds_with_margin"] is not None:
            estimate["fits_time_limit"] = estimate["estimated_seconds_with_margin"] <= limit
    return estimate


def format_grid_estimate(estimate: Mapping[str, Any]) -> str:
    """Return a human-readable report for ``estimate_grid_run`` results."""
    lines = [
        f"Cells to fit: {estimate['fitted_cells']} of {estimate['grid_cells']} ("
        + (f"{estimate['completed_cells']} already complete, " if estimate.get("completed_cells") else "")
        + f"{estimate['skipped_cells']} skipped: no finite data or not selected)",
        f"Ranks: {estimate['ranks']} ({estimate['fitting_workers']} fitting worker"
        f"{'s' if estimate['fitting_workers'] != 1 else ''}"
        f"{'; rank 0 also fits' if estimate['ranks'] == 1 else '; rank 0 writes'})",
        f"Cells per worker: {estimate['min_cells_per_worker']}-{estimate['max_cells_per_worker']}"
        + (f" ({estimate['idle_workers']} workers idle)" if estimate["idle_workers"] else ""),
        *(
            [f"WARNING: {len(estimate['constant_cell_ids'])} cells have constant data and will be "
             f"fitted but are likely to fail or be meaningless (flat IDs: "
             f"{' '.join(map(str, estimate['constant_cell_ids']))})"]
            if estimate["constant_cell_ids"] else []
        ),
        f"Uncompressed output size: {estimate['uncompressed_output_bytes'] / 1e9:.2f} GB "
        "(compression usually makes the file smaller)",
    ]
    if estimate["estimated_seconds"] is None:
        lines.append("Run time: not estimated; pass a measured seconds-per-cell value.")
    else:
        lines.append(
            f"Estimated fitting time: {_format_seconds(estimate['estimated_seconds'])} "
            f"(busiest worker), {_format_seconds(estimate['estimated_seconds_with_margin'])} with safety margin"
        )
    if estimate["time_limit_seconds"] is not None:
        limit = _format_seconds(estimate["time_limit_seconds"])
        if estimate["fits_time_limit"] is None:
            lines.append(f"Time limit {limit}: cannot be checked without seconds per cell.")
        elif estimate["fits_time_limit"]:
            lines.append(f"Time limit {limit}: OK.")
        else:
            lines.append(f"WARNING: time limit {limit} is shorter than the estimate with margin.")
    return "\n".join(lines)


def fit_grid_mpi(
    data: xr.DataArray,
    output: str | os.PathLike[str],
    *,
    uncertainty: xr.DataArray | None = None,
    regressors: xr.DataArray | xr.Dataset | np.ndarray | None = None,
    model: str = "vanilla_ar1",
    grid_dims: Sequence[str] | None = None,
    output_variables: Mapping[str, Literal["single", "double"]] | None = None,
    posterior_means: bool = True,
    overwrite: bool = False,
    resume: bool = False,
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
    unless ``overwrite=True`` or ``resume=True``.

    With ``resume=True``, an existing ``output`` is reused and only cells whose
    ``fit_status`` is not complete (1) are fitted, so a large grid can be split
    across several batch jobs that each stop at a wall-clock limit. Failed cells
    (-1) are retried. The file must have been written with the same model,
    grid, sampler settings, priors, and output variables; otherwise a
    ``ValueError`` explains the mismatch. If ``output`` does not exist, a new
    run starts, so the same command can be used for every submission. Results
    are written one cell at a time, so an interrupted run loses only the cells
    in progress.

    Cells with at least one finite value that still need fitting are dealt
    round-robin, in flat (row-major) order, to ranks ``1..N-1``.

    ``output_variables`` maps posterior variable names to ``"single"`` (float32)
    or ``"double"`` (float64). Only listed posterior variables are saved; ``None``
    saves all in single precision, and an empty mapping saves only coordinates,
    ``fit_status``, and ``divergent_transitions``. This affects storage and MPI
    transfers, not sampling.

    With ``posterior_means=True`` (the default), each non-time-varying parameter
    (for example ``sigma_AR``, ``rhoAR1``, or static ``beta``) also gets a
    ``<name>_mean`` variable over the grid (plus ``regressor`` where applicable),
    holding the double-precision posterior mean for each cell. Means are written
    whether or not that parameter's draws are selected in ``output_variables``.

    The file always contains ``fit_status`` and a per-cell
    ``divergent_transitions`` count, and its global attributes record the model,
    sampler settings, resolved priors, and software versions. Invalid priors are
    rejected during setup, before any fits run.
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
    warmup = int(fit_options.get("warmup", 1000))
    samples = int(fit_options.get("samples", 1000))
    setup_error = None
    output_schema: dict[str, tuple[str, tuple[str, ...]]] = {}

    if rank == 0:
        try:
            output_schema = _output_schema(selected_grid_dims, spec, output_variables)
            prior_values = _resolve_priors(
                spec,
                nreg,
                fit_options.get("priors"),
                {name: fit_options.get(name) for name in _PRIOR_ARGUMENTS},
            )
            metadata = _run_metadata(
                model_name,
                prior_values,
                chains=chains,
                warmup=warmup,
                samples=samples,
                seed=fit_options.get("seed"),
                sampling=fit_options.get("sampling", "monthly"),
                sampler_options=fit_options,
            )
            if overwrite and resume:
                raise ValueError("Set only one of overwrite=True and resume=True.")
            mean_schema = _mean_schema(_posterior_dims(selected_grid_dims, spec)) if posterior_means else {}
            completed = np.zeros(cell_count, dtype=bool)
            resuming = resume and output_path.exists()
            if output_path.exists() and not (overwrite or resume):
                raise FileExistsError(
                    f"Output already exists: {output_path}; set resume=True to continue it "
                    "or overwrite=True to replace it."
                )
            if resuming:
                _check_resumable(
                    output_path, data, selected_grid_dims, nreg, metadata,
                    output_schema, mean_schema, chains, samples,
                )
                completed = _read_completed_cells(output_path, selected_grid_dims)
            has_data = np.isfinite(data.to_numpy().reshape(data.sizes["time"], -1)).any(axis=0)
            pending = np.flatnonzero(has_data & ~completed)
            _get_compiled_model(model_name, spec, fit_options.get("cache_dir"))
            if resuming:
                import netCDF4

                with netCDF4.Dataset(output_path, "r+") as output_file:
                    entry = (
                        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} resumed: "
                        f"{int(completed.sum())} cells complete, {pending.size} to fit"
                    )
                    history = getattr(output_file, "history", "")
                    output_file.history = f"{history}\n{entry}" if history else entry
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists():
                    output_path.unlink()
                _create_result_file(
                    output_path, data, selected_grid_dims, nreg, regressor_names,
                    metadata, output_schema, mean_schema, chains, samples,
                )
        except Exception as exc:
            setup_error = str(exc)
    setup_error = comm.bcast(setup_error, root=0)
    if setup_error:
        raise RuntimeError(f"Could not initialize grid fit: {setup_error}")
    output_schema = comm.bcast(output_schema, root=0)
    pending = comm.bcast(pending if rank == 0 else None, root=0)
    mean_schema = _mean_schema(_posterior_dims(selected_grid_dims, spec)) if posterior_means else {}
    comm.Barrier()

    grid_shape = tuple(data.sizes[dim] for dim in selected_grid_dims)
    if process_count == 1:
        local_indices = pending
    elif rank == 0:
        local_indices = np.empty(0, dtype=int)
    else:
        local_indices = pending[rank - 1 :: process_count - 1]

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
            payload = {
                name: posterior[name].to_numpy().astype(dtype, copy=False)
                for name, (dtype, _) in output_schema.items()
            }
            for name, (source, _) in mean_schema.items():
                payload[name] = posterior[source].mean(("chain", "draw")).to_numpy().astype("f8")
            if "divergent_transitions" in posterior.attrs:
                payload["divergent_transitions"] = np.int32(posterior.attrs["divergent_transitions"])
            return cell_index, payload, None
        except Exception as exc:
            return cell_index, None, str(exc)

    if rank == 0:
        if process_count == 1:
            for flat_index in local_indices:
                _write_result_to_file(output_path, *run_cell(int(flat_index)))
        else:
            for _ in range(len(pending)):
                _write_result_to_file(output_path, *comm.recv(source=MPI.ANY_SOURCE, tag=1))
    else:
        for flat_index in local_indices:
            comm.send(run_cell(int(flat_index)), dest=0, tag=1)

    comm.Barrier()
    if rank == 0:
        return xr.open_dataset(output_path)
    return None
