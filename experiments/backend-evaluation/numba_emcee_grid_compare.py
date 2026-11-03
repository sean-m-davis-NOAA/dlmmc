"""Compare emcee grid posteriors and smoothed-state uncertainty with CmdStan."""

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


PARAMETER_NAMES = ("sigma_trend", "sigma_seas", "sigma_AR", "rhoAR1")
STATE_NAMES = ("trend", "slope", "seasonal", "ar", "beta")
CMDSTAN_PARAMETER_NAMES = {
    "sigma_trend": "sigma_trend",
    "sigma_seas": "sigma_seas",
    "sigma_AR": "sigma_AR",
    "rhoAR1": "rhoAR1",
}


def normalized_rmse(
    values: np.ndarray, reference: np.ndarray, reference_sd: np.ndarray
) -> float:
    rmse = np.sqrt(np.mean(np.square(values - reference)))
    scale = np.sqrt(np.mean(np.square(reference_sd)))
    return float(rmse / scale) if scale else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilots",
        type=Path,
        nargs="+",
        default=[
            Path("/tmp/dlmmc-numba-emcee-grid-mpi-pilot.npz"),
            Path("/tmp/dlmmc-numba-emcee-grid-mpi-repeat-seed20261103.npz"),
        ],
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-torch-real-grid-cmdstan.npz"),
    )
    parser.add_argument(
        "--cmdstan-reference",
        type=Path,
        default=Path("/tmp/dlmmc-cmdstan-multicell-grid.nc"),
    )
    parser.add_argument("--thin", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dlmmc-numba-emcee-grid-comparison.json"),
    )
    args = parser.parse_args()
    if args.thin < 1 or args.batch_size < 1:
        parser.error("--thin and --batch-size must be positive")
    for path in [*args.pilots, args.reference, args.cmdstan_reference]:
        if not path.exists():
            parser.error(f"input not found: {path}")

    jax.config.update("jax_enable_x64", True)
    with np.load(args.reference) as fixed:
        fixed_cells = np.asarray(fixed["cell_indices"])
        series = np.asarray(fixed["series"], dtype=np.float64)
        uncertainty = np.asarray(fixed["stddev"], dtype=np.float64)
        regressors = np.asarray(fixed["regressors"], dtype=np.float64)

    with xr.open_dataset(args.cmdstan_reference) as posterior:
        cmdstan_cells = np.asarray(posterior["cell"].values)
        if not np.array_equal(fixed_cells, cmdstan_cells):
            parser.error("fixed input and CmdStan NetCDF cell coordinates do not match")
        cmdstan_parameters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        cmdstan_parameter_quantiles: dict[str, dict[str, np.ndarray]] = {}
        for name, variable in CMDSTAN_PARAMETER_NAMES.items():
            draws = np.asarray(
                posterior[variable].transpose("cell", "chain", "draw").values
            )
            cmdstan_parameters[name] = (
                draws.mean(axis=(1, 2)),
                draws.std(axis=(1, 2)),
            )
            cmdstan_parameter_quantiles[name] = {
                str(q): np.quantile(draws, q, axis=(1, 2))
                for q in (0.05, 0.5, 0.95)
            }

        cmdstan_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in STATE_NAMES:
            variable = posterior[name]
            if name == "beta":
                draws = np.asarray(
                    variable.transpose("cell", "chain", "draw", "regressor").values
                )
            else:
                draws = np.asarray(
                    variable.transpose("cell", "chain", "draw", "time").values
                )
            cmdstan_states[name] = (
                draws.mean(axis=(1, 2)),
                draws.std(axis=(1, 2)),
            )
        fit_status = np.asarray(posterior["fit_status"].values)

    report: dict[str, object] = {
        "cell_indices": fixed_cells.tolist(),
        "cmdstan_fit_status": fit_status.tolist(),
        "thin": args.thin,
        "runs": [],
    }
    for run_index, path in enumerate(args.pilots):
        with np.load(path) as pilot:
            chains = np.asarray(pilot["chain"], dtype=np.float64)
            pilot_cells = np.asarray(pilot["cell_indices"])
            if not np.array_equal(fixed_cells, pilot_cells):
                parser.error(f"emcee cells in {path} do not match the fixed reference")
            effective_samples = np.asarray(pilot["effective_samples"], dtype=np.float64)
            total_seconds = float(pilot["total_seconds"])
            acceptance = np.asarray(pilot["acceptance_fraction"], dtype=np.float64)

        run: dict[str, object] = {
            "pilot": str(path),
            "chain_shape": list(chains.shape),
            "total_wall_seconds": total_seconds,
            "acceptance_mean": float(acceptance.mean()),
            "acceptance_by_cell": acceptance.mean(axis=1).tolist(),
            "emcee_ess_min_by_parameter": {
                name: float(effective_samples[:, index].min())
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "emcee_ess_sum_per_wall_second_by_parameter": {
                name: float(effective_samples[:, index].sum() / total_seconds)
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "parameter_comparison": {},
        }
        for parameter_index, name in enumerate(PARAMETER_NAMES):
            draws = chains[..., parameter_index].reshape(chains.shape[0], -1)
            reference_mean, reference_sd = cmdstan_parameters[name]
            run["parameter_comparison"][name] = {
                "mean_rmse_over_cmdstan_sd": normalized_rmse(
                    draws.mean(axis=1), reference_mean, reference_sd
                ),
                "sd_rmse_over_cmdstan_sd": normalized_rmse(
                    draws.std(axis=1), reference_sd, reference_sd
                ),
                "quantile_rmse_over_cmdstan_sd": {
                    str(q): normalized_rmse(
                        np.quantile(draws, q, axis=1),
                        cmdstan_parameter_quantiles[name][str(q)],
                        reference_sd,
                    )
                    for q in (0.05, 0.5, 0.95)
                },
            }

        run_started = time.perf_counter()
        cell_state_results: list[dict[str, object]] = []
        state_mean_errors = {name: [] for name in STATE_NAMES}
        state_sd_errors = {name: [] for name in STATE_NAMES}
        for cell_position, cell_index in enumerate(fixed_cells):
            parameters = chains[cell_position, :: args.thin].reshape(
                -1, chains.shape[-1]
            )
            smoothed = draw_smoothed_states(
                parameters,
                jnp.asarray(series[cell_position], dtype=jnp.float64),
                jnp.asarray(uncertainty[cell_position], dtype=jnp.float64),
                jnp.asarray(regressors[cell_position], dtype=jnp.float64),
                args.seed + run_index * 1000 + int(cell_index),
                args.batch_size,
            )
            cell_result: dict[str, object] = {
                "cell_index": int(cell_index),
                "smoother_draws": int(len(parameters)),
                "state_comparison": {},
            }
            for name in STATE_NAMES:
                samples = smoothed[name]
                mean, sd = cmdstan_states[name]
                mean_error = normalized_rmse(
                    samples.mean(axis=0), mean[cell_position], sd[cell_position]
                )
                sd_error = normalized_rmse(
                    samples.std(axis=0), sd[cell_position], sd[cell_position]
                )
                state_mean_errors[name].append(mean_error)
                state_sd_errors[name].append(sd_error)
                cell_result["state_comparison"][name] = {
                    "mean_rmse_over_cmdstan_sd": mean_error,
                    "sd_rmse_over_cmdstan_sd": sd_error,
                }
            cell_state_results.append(cell_result)

        run["state_mean_rmse_over_cmdstan_sd_by_state"] = {
            name: float(np.mean(values)) for name, values in state_mean_errors.items()
        }
        run["state_mean_max_cell_rmse_over_cmdstan_sd_by_state"] = {
            name: float(np.max(values)) for name, values in state_mean_errors.items()
        }
        run["state_sd_rmse_over_cmdstan_sd_by_state"] = {
            name: float(np.mean(values)) for name, values in state_sd_errors.items()
        }
        run["state_comparison_by_cell"] = cell_state_results
        run["state_postprocessing_seconds"] = time.perf_counter() - run_started
        report["runs"].append(run)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for run in report["runs"]:
        print(f"pilot={run['pilot']}")
        print(f"wall_seconds={run['total_wall_seconds']:.3f}")
        print(f"acceptance_mean={run['acceptance_mean']:.6g}")
        print(f"ess_per_wall_second={run['emcee_ess_sum_per_wall_second_by_parameter']}")
        print(f"parameter_comparison={run['parameter_comparison']}")
        print(f"state_mean_error={run['state_mean_rmse_over_cmdstan_sd_by_state']}")
        print(f"state_sd_error={run['state_sd_rmse_over_cmdstan_sd_by_state']}")
    print(f"saved_comparison={args.output}")


if __name__ == "__main__":
    main()
