"""Run independent CmdStanPy posterior fits for cells in a saved grid reference."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from cmdstanpy import CmdStanModel

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)

import dlmmc


STATE_NAMES = ("trend", "slope", "seasonal", "ar")
PARAMETER_NAMES = ("sigma_trend", "sigma_seas", "sigma_AR", "rhoAR1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-multicell-stan-reference.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dlmmc-cmdstan-multicell-posterior.npz"),
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--parallel-chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if args.chains < 1 or not 1 <= args.parallel_chains <= args.chains:
        parser.error("require chains >= 1 and 1 <= parallel-chains <= chains")
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be non-negative and samples must be positive")
    if not args.reference.exists():
        parser.error(f"grid reference not found: {args.reference}")

    reference = np.load(args.reference)
    cell_count = reference["series"].shape[0]
    data_count = reference["series"].shape[1]
    regressor_count = reference["regressors"].shape[2]
    executable = dlmmc.compile_models(["vanilla_ar1"])["vanilla_ar1"]
    model = CmdStanModel(exe_file=executable)

    parameter_draws: dict[str, list[np.ndarray]] = {name: [] for name in PARAMETER_NAMES}
    parameter_draws["beta"] = []
    state_means: dict[str, list[np.ndarray]] = {name: [] for name in STATE_NAMES}
    state_sds: dict[str, list[np.ndarray]] = {name: [] for name in STATE_NAMES}
    cell_fit_seconds: list[float] = []
    overall_started = time.perf_counter()

    for cell_index in range(cell_count):
        data = {
            "N": data_count,
            "nreg": regressor_count,
            "sampling": 1.0,
            "time_series": reference["series"][cell_index],
            "stddev": reference["stddev"][cell_index],
            "regressors": reference["regressors"][cell_index],
            "sigma_trend_prior": 1e-4,
            "sigma_seas_prior": 0.01,
            "sigma_AR_prior": 0.5,
            "S": 10.0,
        }
        started = time.perf_counter()
        fit = model.sample(
            data=data,
            chains=args.chains,
            parallel_chains=args.parallel_chains,
            iter_warmup=args.warmup,
            iter_sampling=args.samples,
            seed=args.seed + cell_index,
            show_progress=False,
        )
        draws = fit.draws_xr(vars=[*PARAMETER_NAMES, *STATE_NAMES, "beta"])
        cell_fit_seconds.append(time.perf_counter() - started)
        for name in PARAMETER_NAMES + ("beta",):
            parameter_draws[name].append(np.asarray(draws[name].to_numpy()))
        for name in STATE_NAMES:
            values = np.asarray(draws[name].to_numpy())
            state_means[name].append(values.mean(axis=(0, 1)))
            state_sds[name].append(values.std(axis=(0, 1)))
        print(
            f"cell={int(reference['cell_indices'][cell_index])} "
            f"elapsed_seconds={cell_fit_seconds[-1]:.3f} "
            f"divergences={fit.divergences}"
        )

    total_seconds = time.perf_counter() - overall_started
    result: dict[str, np.ndarray] = {
        "cell_indices": reference["cell_indices"],
        "cell_fit_seconds": np.asarray(cell_fit_seconds),
        "total_seconds": np.asarray(total_seconds),
        "chains": np.asarray(args.chains),
        "warmup": np.asarray(args.warmup),
        "samples": np.asarray(args.samples),
        "seed": np.asarray(args.seed),
    }
    for name, cells in parameter_draws.items():
        result[f"draws_{name}"] = np.stack(cells)
    for name in STATE_NAMES:
        result[f"mean_{name}"] = np.stack(state_means[name])
        result[f"sd_{name}"] = np.stack(state_sds[name])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"cmdstan_grid_cells={cell_count}")
    print(f"cmdstan_grid_total_seconds={total_seconds:.3f}")
    print(f"cmdstan_grid_fit_seconds_median={float(np.median(cell_fit_seconds)):.3f}")
    print(f"saved_cmdstan_grid={args.output}")


if __name__ == "__main__":
    main()
