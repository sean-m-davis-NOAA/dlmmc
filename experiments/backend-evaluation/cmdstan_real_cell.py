"""Fit the benchmark ozone cell with CmdStanPy and save data/draws for PyMC comparison."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
from cmdstanpy import CmdStanModel

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

    fixed_parameters = {
        "sigma_trend": 7.732e-5,
        "sigma_seas": 0.001501,
        "sigma_AR": 0.01757,
        "rhoAR1": 0.2448,
    }
    prior_scales = {
        "sigma_trend": 1e-4,
        "sigma_seas": 0.01,
        "sigma_AR": 0.5,
    }
    raw_series = np.asarray(data.to_numpy(), dtype=np.float64)
    raw_uncertainty = np.asarray(uncertainty.to_numpy(), dtype=np.float64)
    valid_data = np.isfinite(raw_series)
    filled_series = np.where(valid_data, raw_series, raw_series[valid_data].mean())
    missing = ~valid_data | ~np.isfinite(raw_uncertainty)
    stan_uncertainty = np.where(missing, 1e20, raw_uncertainty)
    regressor_names = list(regressors.data_vars)
    regressor_values = np.column_stack(
        [np.asarray(regressors[name].to_numpy(), dtype=np.float64) for name in regressor_names]
    )
    stan_executable = dlmmc.compile_models(["vanilla_ar1"])["vanilla_ar1"]
    stan_model = CmdStanModel(exe_file=stan_executable)
    diagnostic = stan_model.log_prob(
        fixed_parameters,
        {
            "N": len(filled_series),
            "nreg": regressor_values.shape[1],
            "sampling": 1.0,
            "time_series": filled_series,
            "stddev": stan_uncertainty,
            "regressors": regressor_values,
            "sigma_trend_prior": prior_scales["sigma_trend"],
            "sigma_seas_prior": prior_scales["sigma_seas"],
            "sigma_AR_prior": prior_scales["sigma_AR"],
            "S": 10.0,
        },
        jacobian=False,
        sig_figs=16,
    ).iloc[0]
    prior_parameter_term = sum(
        -0.5 * (fixed_parameters[name] / scale) ** 2
        for name, scale in prior_scales.items()
    )
    prior_gradient = np.array(
        [
            -fixed_parameters["sigma_trend"] / prior_scales["sigma_trend"] ** 2,
            -fixed_parameters["sigma_seas"] / prior_scales["sigma_seas"] ** 2,
            -fixed_parameters["sigma_AR"] / prior_scales["sigma_AR"] ** 2,
            0.0,
        ]
    )
    transform_derivative = np.array(
        [
            fixed_parameters["sigma_trend"],
            fixed_parameters["sigma_seas"],
            fixed_parameters["sigma_AR"],
            fixed_parameters["rhoAR1"] * (1.0 - fixed_parameters["rhoAR1"]),
        ]
    )
    fixed_gradient = np.array(
        [
            diagnostic["g_sigma_trend"],
            diagnostic["g_sigma_seas"],
            diagnostic["g_sigma_AR"],
            diagnostic["g_rhoAR1"],
        ]
    ) / transform_derivative - prior_gradient
    fixed_log_likelihood = float(diagnostic["lp__"]) - prior_parameter_term
    print(f"cmdstan_fixed_log_likelihood={fixed_log_likelihood:.12g}")
    print(f"cmdstan_fixed_likelihood_gradient={fixed_gradient.tolist()}")

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
        fixed_parameters=np.array(list(fixed_parameters.values())),
        fixed_log_likelihood=fixed_log_likelihood,
        fixed_likelihood_gradient=fixed_gradient,
        fit_seconds=fit_seconds,
        divergences=posterior.attrs.get("divergent_transitions", -1),
    )
    print(f"saved_reference={args.output}")


if __name__ == "__main__":
    main()
