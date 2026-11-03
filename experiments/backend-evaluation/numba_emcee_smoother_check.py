"""Check smoothed-state summaries for saved Numba-emcee draws.

The Numba/emcee sampler remains separate; this uses the existing JAX CPU
simulation smoother only as a post-processing parity check.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from numpyro_real_cell import draw_smoothed_states


STATE_NAMES = ("trend", "slope", "seasonal", "ar", "beta")
PARAMETER_NAMES = ("sigma_trend", "sigma_seas", "sigma_AR", "rhoAR1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmdstan-reference",
        type=Path,
        default=Path("/tmp/dlmmc-backend-eval-real-cell-cmdstan-smoother.npz"),
    )
    parser.add_argument(
        "--emcee-runs",
        type=Path,
        nargs="+",
        default=[
            Path("/tmp/dlmmc-numba-emcee-real-cell.npz"),
            Path("/tmp/dlmmc-numba-emcee-real-cell-repeat2.npz"),
        ],
    )
    parser.add_argument("--thin", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    if args.thin < 1 or args.batch_size < 1:
        parser.error("--thin and --batch-size must be positive")
    if not args.cmdstan_reference.exists():
        parser.error(f"CmdStan reference not found: {args.cmdstan_reference}")
    missing = [path for path in args.emcee_runs if not path.exists()]
    if missing:
        parser.error(f"emcee output not found: {missing[0]}")

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    reference = np.load(args.cmdstan_reference)
    series = jax.numpy.asarray(reference["series"], dtype=jax.numpy.float64)
    uncertainty = jax.numpy.asarray(reference["uncertainty"], dtype=jax.numpy.float64)
    regressors = jax.numpy.asarray(reference["regressors"], dtype=jax.numpy.float64)
    cmdstan_state_means = {
        name: np.asarray(reference[name]).mean(axis=(0, 1)) for name in STATE_NAMES
    }
    cmdstan_state_sds = {
        name: np.asarray(reference[name]).std(axis=(0, 1)) for name in STATE_NAMES
    }

    summaries: list[dict[str, object]] = []
    for run_index, path in enumerate(args.emcee_runs):
        with np.load(path) as saved:
            chain = np.asarray(saved["chain"], dtype=np.float64)
            acceptance = float(np.asarray(saved["acceptance_fraction"]).mean())
            autocorrelation = np.asarray(saved["autocorrelation_time"])
        selected = chain[:: args.thin].reshape(-1, chain.shape[-1])
        started = time.perf_counter()
        smoothed = draw_smoothed_states(
            selected,
            series,
            uncertainty,
            regressors,
            args.seed + run_index,
            args.batch_size,
        )
        smoother_seconds = time.perf_counter() - started

        metrics: dict[str, object] = {
            "run": str(path),
            "emcee_draws": int(selected.shape[0]),
            "acceptance_fraction": acceptance,
            "autocorrelation_time": autocorrelation.tolist(),
            "smoother_seconds": smoother_seconds,
        }
        for name in STATE_NAMES:
            samples = smoothed[name]
            mean = samples.mean(axis=0)
            rmse = float(np.sqrt(np.mean(np.square(mean - cmdstan_state_means[name]))))
            reference_sd = float(np.sqrt(np.mean(np.square(cmdstan_state_sds[name]))))
            normalized_rmse = rmse / reference_sd if reference_sd else float("nan")
            metrics[f"{name}_posterior_mean_rmse"] = rmse
            metrics[f"{name}_rmse_over_cmdstan_posterior_sd"] = normalized_rmse
            print(
                f"run={run_index + 1} state={name} draws={len(samples)} "
                f"mean_rmse={rmse:.6g} rmse_over_cmdstan_sd={normalized_rmse:.6g}"
            )
        summaries.append(metrics)

    output = args.cmdstan_reference.with_name("numba-emcee-smoother-check.json")
    output.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(f"saved_state_comparison={output}")


if __name__ == "__main__":
    main()
