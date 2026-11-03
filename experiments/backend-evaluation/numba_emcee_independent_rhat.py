"""Compute R-hat from independent emcee ensembles, not dependent walkers.

One prespecified walker trajectory from each independently seeded ensemble is
used as one chain. This is a conservative diagnostic approximation: walkers
within each ensemble remain coupled through the stretch move.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr


PARAMETER_NAMES = ("sigma_trend", "sigma_seas", "sigma_AR", "rhoAR1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--walker-index",
        type=int,
        default=0,
        help="Prespecified walker trajectory to use from each independent ensemble.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dlmmc-numba-emcee-independent-rhat.json"),
    )
    args = parser.parse_args()
    if len(args.runs) < 4:
        parser.error("at least four independent emcee runs are required")
    missing = [path for path in args.runs if not path.exists()]
    if missing:
        parser.error(f"run archive not found: {missing[0]}")

    chains: list[np.ndarray] = []
    cell_indices: np.ndarray | None = None
    step_count: int | None = None
    for path in args.runs:
        with np.load(path) as saved:
            chain = np.asarray(saved["chain"], dtype=np.float64)
            run_cells = np.asarray(saved["cell_indices"])
        if chain.ndim != 4 or chain.shape[-1] != len(PARAMETER_NAMES):
            parser.error(f"unexpected chain shape in {path}: {chain.shape}")
        if not 0 <= args.walker_index < chain.shape[2]:
            parser.error(
                f"walker index {args.walker_index} is out of range for {path}"
            )
        if cell_indices is None:
            cell_indices = run_cells
            step_count = chain.shape[1]
        elif not np.array_equal(cell_indices, run_cells):
            parser.error(f"cell IDs in {path} do not match the first archive")
        elif chain.shape[1] != step_count:
            parser.error(f"sample count in {path} does not match the first archive")
        chains.append(
            np.transpose(chain[:, :, args.walker_index, :], (1, 0, 2))
        )

    assert cell_indices is not None
    chain_array = np.stack(chains, axis=0)
    draws = xr.DataArray(
        chain_array,
        dims=("chain", "draw", "cell", "parameter"),
        coords={
            "cell": cell_indices,
            "parameter": np.asarray(PARAMETER_NAMES),
        },
    )
    rhat = az.rhat(draws, method="rank")
    bulk_ess = az.ess(draws, method="bulk")

    result: dict[str, object] = {
        "runs": [str(path) for path in args.runs],
        "independent_ensemble_count": len(args.runs),
        "walker_index_per_ensemble": args.walker_index,
        "draws_per_chain": int(step_count or 0),
        "cell_indices": cell_indices.tolist(),
        "parameters": {},
    }
    for name in PARAMETER_NAMES:
        parameter_rhat = np.asarray(rhat.sel(parameter=name).values)
        parameter_ess = np.asarray(bulk_ess.sel(parameter=name).values)
        worst_rhat_index = int(np.nanargmax(parameter_rhat))
        lowest_ess_index = int(np.nanargmin(parameter_ess))
        result["parameters"][name] = {
            "rhat_max": float(parameter_rhat[worst_rhat_index]),
            "rhat_max_cell": int(cell_indices[worst_rhat_index]),
            "rhat_median": float(np.nanmedian(parameter_rhat)),
            "bulk_ess_min": float(parameter_ess[lowest_ess_index]),
            "bulk_ess_min_cell": int(cell_indices[lowest_ess_index]),
            "bulk_ess_median": float(np.nanmedian(parameter_ess)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["parameters"], indent=2, sort_keys=True))
    print(f"saved_independent_ensemble_rhat={args.output}")


if __name__ == "__main__":
    main()
