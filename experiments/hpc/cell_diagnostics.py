"""Write per-cell R-hat and ESS rows for a dlmmc MPI grid NetCDF.

Rows follow ``hera_cpu_mpi_cell_diagnostics.csv.example``. Diagnostics are
computed by ArviZ from the stored posterior draws over every scalar and
regressor parameter, so the worst-mixing parameter in each cell is reported.
Divergence counts come from the per-cell ``divergent_transitions`` variable;
grid files written before dlmmc stored it leave the column blank.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import xarray as xr

FIELDS = [
    "run_id",
    "flat_cell_id",
    "pressure_index",
    "latitude_index",
    "fit_status",
    "divergences",
    "max_rhat",
    "min_ess_bulk",
    "min_ess_tail",
    "diagnostic_source",
    "notes",
]
SAMPLE_DIMS = ("chain", "draw")
GRID_DIMS = ("pressure", "latitude")


def _parameter_names(ds: xr.Dataset) -> list[str]:
    """Return non-time-varying posterior parameters; latent states are skipped."""
    names = []
    for name, var in ds.data_vars.items():
        dims = set(var.dims)
        if set(SAMPLE_DIMS) <= dims and set(GRID_DIMS) <= dims and "time" not in dims:
            names.append(name)
    return names


def cell_diagnostics(ds: xr.Dataset, run_id: str) -> list[dict[str, object]]:
    import arviz as az

    nchains = ds.sizes["chain"]
    names = _parameter_names(ds)
    if not names:
        raise ValueError("No posterior parameters with chain/draw/pressure/latitude dims")

    fitted = ds["fit_status"].values == 1
    rhat_parts, bulk_parts, tail_parts = [], [], []
    for name in names:
        var = ds[name].where(xr.DataArray(fitted, dims=GRID_DIMS))
        extra = [d for d in var.dims if d not in SAMPLE_DIMS + GRID_DIMS]
        rhat = az.rhat(var, sample_dims=list(SAMPLE_DIMS))
        bulk = az.ess(var, sample_dims=list(SAMPLE_DIMS), method="bulk")
        tail = az.ess(var, sample_dims=list(SAMPLE_DIMS), method="tail", prob=(0.05, 0.95))
        if extra:
            rhat, bulk, tail = rhat.max(extra), bulk.min(extra), tail.min(extra)
        rhat_parts.append(rhat.transpose(*GRID_DIMS).values)
        bulk_parts.append(bulk.transpose(*GRID_DIMS).values)
        tail_parts.append(tail.transpose(*GRID_DIMS).values)

    max_rhat = np.max(rhat_parts, axis=0)
    min_bulk = np.min(bulk_parts, axis=0)
    min_tail = np.min(tail_parts, axis=0)
    nlat = ds.sizes["latitude"]
    if "divergent_transitions" in ds:
        divergences = ds["divergent_transitions"].transpose(*GRID_DIMS).values
        note = ""
    else:
        divergences = None
        note = "divergences not stored in grid NetCDF"
    if nchains < 2:
        note = "; ".join(filter(None, [note, "R-hat requires at least 2 chains"]))

    rows = []
    for p, l in zip(*np.nonzero(fitted)):
        rows.append(
            {
                "run_id": run_id,
                "flat_cell_id": int(p * nlat + l),
                "pressure_index": int(p),
                "latitude_index": int(l),
                "fit_status": 1,
                "divergences": "" if divergences is None or not np.isfinite(divergences[p, l])
                else int(divergences[p, l]),
                "max_rhat": "" if nchains < 2 else f"{max_rhat[p, l]:.4f}",
                "min_ess_bulk": f"{min_bulk[p, l]:.1f}",
                "min_ess_tail": f"{min_tail[p, l]:.1f}",
                "diagnostic_source": "arviz on NetCDF draws; params " + "|".join(names),
                "notes": note,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("netcdf", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, help="CSV path; defaults to stdout")
    args = parser.parse_args(argv)

    with xr.open_dataset(args.netcdf) as ds:
        rows = cell_diagnostics(ds, args.run_id)

    handle = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
