"""Fit the real-cell comparison model with PyMC StateSpace and nutpie."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc_extras.statespace import structural as st
from pymc_extras.statespace.core.properties import Data
from pymc_extras.statespace.models.structural.core import Component


class KnownObservationCovariance(Component):
    """Measurement component for known, time-varying observation variance."""

    def __init__(self) -> None:
        super().__init__(
            name="known_observation_covariance",
            k_endog=1,
            k_states=0,
            k_posdef=0,
            base_observed_state_names=["data"],
            measurement_error=True,
            combine_hidden_states=False,
        )

    def set_data_info(self) -> Data:
        return Data(name="known_obs_cov", shape=(None, 1, 1), is_exogenous=True)

    def set_parameters(self) -> None:
        return None

    def make_symbolic_graph(self) -> None:
        observation_covariance = self.make_and_register_data(
            "known_obs_cov", shape=(None, 1, 1)
        )
        self.ssm["obs_cov"] = observation_covariance
        self.ssm.declare_time_varying("obs_cov")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-backend-eval-real-cell-cmdstan.npz"),
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dlmmc-backend-eval-real-cell-pymc.npz"),
    )
    args = parser.parse_args()

    reference = np.load(args.reference)
    series = reference["series"].astype(np.float64)
    measurement_sd = reference["uncertainty"].astype(np.float64)
    regression_values = reference["regressors"].astype(np.float64)
    regression_names = reference["regressor_names"].tolist()
    series_range = float(np.ptp(series))
    if not np.isfinite(series_range) or series_range <= 0:
        raise ValueError("Series must have a finite, non-zero range.")

    scaled_series = (series - float(series.mean())) / series_range
    scaled_measurement_sd = measurement_sd / series_range

    # The leading row is the unobserved prior-time state. The first scored
    # observation uses regressor[0], matching Stan's F[t] indexing.
    observed = np.concatenate([[np.nan], scaled_series])[:, None]
    known_obs_cov = np.concatenate([[1.0], scaled_measurement_sd**2])[:, None, None]
    regression_data = np.vstack([np.zeros((1, len(regression_names))), regression_values])

    model_ss = (
        st.LevelTrend(order=2, innovations_order=[0, 1], name="trend")
        + st.FrequencySeasonality(
            season_length=12, n=2, innovations=True, name="season"
        )
        + st.Regression(
            state_names=regression_names,
            name="regression",
            innovations=False,
        )
        + st.Autoregressive(order=1, name="ar")
        + KnownObservationCovariance()
    ).build(verbose=False, cov_jitter=0.0)

    state_count = len(regression_names) + 7
    with pm.Model(coords=model_ss.coords) as model:
        pm.Normal(
            "initial_trend",
            mu=0,
            sigma=np.sqrt(10.0),
            dims=model_ss.param_dims["initial_trend"],
        )
        pm.HalfNormal(
            "sigma_trend", sigma=1e-4, dims=model_ss.param_dims["sigma_trend"]
        )
        pm.Normal(
            "params_season",
            mu=0,
            sigma=np.sqrt(10.0),
            dims=model_ss.param_dims["params_season"],
        )
        pm.HalfNormal("sigma_season", sigma=0.01)
        rho = pm.Uniform(
            "params_ar",
            lower=0.0,
            upper=1.0,
            dims=model_ss.param_dims["params_ar"],
        )
        sigma_ar = pm.HalfNormal("sigma_ar", sigma=0.5)
        initial_covariance_diagonal = pt.concatenate(
            [
                pt.zeros((state_count - 1,)),
                pt.stack([sigma_ar**2 / (1.0 - rho[0] ** 2)]),
            ]
        )
        pm.Deterministic(
            "P0",
            pt.diag(initial_covariance_diagonal),
            dims=model_ss.param_dims["P0"],
        )
        pm.Data("known_obs_cov", known_obs_cov)
        pm.Normal(
            "beta_regression",
            mu=0,
            sigma=np.sqrt(10.0),
            dims=model_ss.param_dims["beta_regression"],
        )
        pm.Data("data_regression", regression_data)
        model_ss.build_statespace_graph(observed)

        started = time.perf_counter()
        trace = pm.sample(
            draws=args.draws,
            tune=args.warmup,
            chains=args.chains,
            cores=1,
            nuts_sampler="nutpie",
            backend="jax",
            compile_kwargs={"gradient_backend": "jax"},
            random_seed=[args.seed + chain for chain in range(args.chains)],
            progressbar=False,
            compute_convergence_checks=True,
        )
        fit_seconds = time.perf_counter() - started

    trace.posterior["beta_regression_physical"] = (
        trace.posterior["beta_regression"] * series_range
    )
    summary_names = [
        "sigma_trend",
        "sigma_season",
        "sigma_ar",
        "params_ar",
        "beta_regression_physical",
    ]
    print(f"pymc_fit_seconds={fit_seconds:.3f}")
    print(az.summary(trace, var_names=summary_names, kind="all").to_string())
    divergences = int(trace.sample_stats["diverging"].to_numpy().sum())
    print(f"divergences={divergences}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sigma_trend=np.asarray(trace.posterior["sigma_trend"]),
        sigma_season=np.asarray(trace.posterior["sigma_season"]),
        sigma_ar=np.asarray(trace.posterior["sigma_ar"]),
        rho=np.asarray(trace.posterior["params_ar"]),
        beta=np.asarray(trace.posterior["beta_regression_physical"]),
        fit_seconds=fit_seconds,
        divergences=divergences,
    )
    print(f"saved_posterior={args.output}")


if __name__ == "__main__":
    main()
