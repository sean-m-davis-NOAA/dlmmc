"""Fit the benchmark ozone cell with CmdStanPy and save data/draws for PyMC comparison."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)

import dlmmc
from dlm_benchmark import load_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure", type=int, default=22)
    parser.add_argument("--latitude", type=int, default=8)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dlmmc-backend-eval-real-cell-cmdstan.npz"),
    )
    args = parser.parse_args()

    data, uncertainty, regressors = load_example(args.pressure, args.latitude)
    started = time.perf_counter()
    posterior = dlmmc.fit(
        data,
        regressors=regressors,
        uncertainty=uncertainty,
        model="vanilla_ar1",
        chains=args.chains,
        warmup=args.warmup,
        samples=args.samples,
        seed=args.seed,
        parallel_chains=args.chains,
        show_progress=False,
    )
    fit_seconds = time.perf_counter() - started

    variables = ["sigma_trend", "sigma_seas", "sigma_AR", "rhoAR1", "beta"]
    print(f"cmdstan_fit_seconds={fit_seconds:.3f}")
    print(az.summary(posterior, var_names=variables, kind="all").to_string())
    print(f"divergences={posterior.attrs.get('divergent_transitions')}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        series=np.asarray(data.to_numpy(), dtype=np.float64),
        uncertainty=np.asarray(uncertainty.to_numpy(), dtype=np.float64),
        regressors=np.column_stack(
            [np.asarray(regressors[name].to_numpy(), dtype=np.float64) for name in regressors.data_vars]
        ),
        regressor_names=np.asarray(list(regressors.data_vars)),
        sigma_trend=posterior["sigma_trend"].to_numpy(),
        sigma_seas=posterior["sigma_seas"].to_numpy(),
        sigma_AR=posterior["sigma_AR"].to_numpy(),
        rhoAR1=posterior["rhoAR1"].to_numpy(),
        beta=posterior["beta"].to_numpy(),
        fit_seconds=fit_seconds,
        divergences=posterior.attrs.get("divergent_transitions", -1),
    )
    print(f"saved_reference={args.output}")


if __name__ == "__main__":
    main()
