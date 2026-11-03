"""Compare Numba/emcee grid draws with CmdStan smoothed-state summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from numpyro_real_cell import draw_smoothed_states


STATE_NAMES = ("trend", "slope", "seasonal", "ar", "beta")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("/tmp/dlmmc-numba-emcee-grid-mpi-pilot.npz"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-torch-real-grid-cmdstan.npz"),
    )
    parser.add_argument(
        "--cmdstan-reference",
        type=Path,
        default=Path("/tmp/dlmmc-cmdstan-multicell-grid-pc4.nc"),
    )
    parser.add_argument("--thin", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dlmmc-numba-emcee-grid-smoother-check.json"),
    )
    args = parser.parse_args()
    if args.thin < 1 or args.batch_size < 1:
        parser.error("--thin and --batch-size must be positive")
    for path in (args.pilot, args.reference, args.cmdstan_reference):
        if not path.exists():
            parser.error(f"input not found: {path}")

    jax.config.update("jax_enable_x64", True)
    with np.load(args.pilot) as pilot, np.load(args.reference) as fixed:
        chains = np.asarray(pilot["chain"], dtype=np.float64)
        pilot_cells = np.asarray(pilot["cell_indices"])
        fixed_cells = np.asarray(fixed["cell_indices"])
        if not np.array_equal(pilot_cells, fixed_cells):
            parser.error("emcee pilot and fixed reference cell indices do not match")
        series = np.asarray(fixed["series"], dtype=np.float64)
        uncertainty = np.asarray(fixed["stddev"], dtype=np.float64)
        regressors = np.asarray(fixed["regressors"], dtype=np.float64)

    with xr.open_dataset(args.cmdstan_reference) as posterior:
        if posterior.sizes.get("cell") != len(pilot_cells):
            parser.error("CmdStan posterior cell dimension does not match the pilot")
        cmdstan_means: dict[str, np.ndarray] = {}
        cmdstan_sds: dict[str, np.ndarray] = {}
        for name in STATE_NAMES:
            samples = np.asarray(posterior[name])
            if name == "beta":
                cmdstan_means[name] = samples.mean(axis=(0, 1)).T
                cmdstan_sds[name] = samples.std(axis=(0, 1)).T
            else:
                cmdstan_means[name] = samples.mean(axis=(0, 1)).T
                cmdstan_sds[name] = samples.std(axis=(0, 1)).T

    started = time.perf_counter()
    cell_results: list[dict[str, object]] = []
    all_ratios: dict[str, list[float]] = {name: [] for name in STATE_NAMES}
    for cell_position, cell_index in enumerate(pilot_cells):
        parameters = chains[cell_position, :: args.thin].reshape(-1, chains.shape[-1])
        smoothed = draw_smoothed_states(
            parameters,
            jnp.asarray(series[cell_position], dtype=jnp.float64),
            jnp.asarray(uncertainty[cell_position], dtype=jnp.float64),
            jnp.asarray(regressors[cell_position], dtype=jnp.float64),
            args.seed + int(cell_index),
            args.batch_size,
        )
        metrics: dict[str, object] = {
            "cell_index": int(cell_index),
            "emcee_draws": int(len(parameters)),
        }
        for name in STATE_NAMES:
            mean = smoothed[name].mean(axis=0)
            delta = mean - cmdstan_means[name][cell_position]
            rmse = float(np.sqrt(np.mean(np.square(delta))))
            reference_sd = cmdstan_sds[name][cell_position]
            scale = float(np.sqrt(np.mean(np.square(reference_sd))))
            normalized_rmse = rmse / scale if scale else float("nan")
            metrics[f"{name}_mean_rmse"] = rmse
            metrics[f"{name}_rmse_over_cmdstan_sd"] = normalized_rmse
            all_ratios[name].append(normalized_rmse)
        cell_results.append(metrics)

    summary = {
        "pilot": str(args.pilot),
        "cmdstan_reference": str(args.cmdstan_reference),
        "thin": args.thin,
        "wall_seconds": time.perf_counter() - started,
        "cell_results": cell_results,
        "mean_rmse_over_cmdstan_sd_by_state": {
            name: float(np.mean(values)) for name, values in all_ratios.items()
        },
        "max_rmse_over_cmdstan_sd_by_state": {
            name: float(np.max(values)) for name, values in all_ratios.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for name in STATE_NAMES:
        print(
            f"{name}_mean_rmse_over_cmdstan_sd="
            f"{summary['mean_rmse_over_cmdstan_sd_by_state'][name]:.6g} "
            f"max_cell={summary['max_rmse_over_cmdstan_sd_by_state'][name]:.6g}"
        )
    print(f"grid_smoother_check_seconds={summary['wall_seconds']:.3f}")
    print(f"saved_grid_smoother_check={args.output}")


if __name__ == "__main__":
    main()
