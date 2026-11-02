"""Compare a CPU NumPyro NUTS fit with the CmdStan real-cell reference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS


PRIOR_SCALES = {
    "sigma_trend": 1e-4,
    "sigma_seas": 0.01,
    "sigma_AR": 0.5,
}
PARAMETER_NAMES = ("sigma_trend", "sigma_seas", "sigma_AR", "rhoAR1")


def kalman_log_likelihood(
    parameters: jax.Array,
    series: jax.Array,
    uncertainty: jax.Array,
    regressors: jax.Array,
) -> jax.Array:
    """Marginal likelihood using the vanilla_ar1 Stan Kalman recursion."""
    sigma_trend, sigma_seas, sigma_ar, rho = parameters
    valid = jnp.isfinite(series)
    series_mean = jnp.sum(jnp.where(valid, series, 0.0)) / jnp.sum(valid)
    filled_series = jnp.where(valid, series, series_mean)
    missing = ~valid | ~jnp.isfinite(uncertainty)
    filled_uncertainty = jnp.where(missing, 1e20, uncertainty)
    data_range = jnp.max(filled_series) - jnp.min(filled_series)
    normalized_series = (filled_series - jnp.mean(filled_series)) / data_range
    normalized_uncertainty = filled_uncertainty / data_range

    time_count, regressor_count = regressors.shape
    state_count = regressor_count + 7
    dtype = parameters.dtype

    transition = jnp.zeros((state_count, state_count), dtype=dtype)
    transition = transition.at[0, 0].set(1.0).at[0, 1].set(1.0)
    transition = transition.at[1, 1].set(1.0)
    for start, angle in ((2, 2.0 * jnp.pi / 12.0), (4, 2.0 * jnp.pi / 6.0)):
        transition = transition.at[start, start].set(jnp.cos(angle))
        transition = transition.at[start, start + 1].set(jnp.sin(angle))
        transition = transition.at[start + 1, start].set(-jnp.sin(angle))
        transition = transition.at[start + 1, start + 1].set(jnp.cos(angle))
    if regressor_count:
        transition = transition.at[6 : 6 + regressor_count, 6 : 6 + regressor_count].set(
            jnp.eye(regressor_count, dtype=dtype)
        )
    transition = transition.at[-1, -1].set(rho)

    process_variance = jnp.concatenate(
        (
            jnp.array([1e-10], dtype=dtype),
            jnp.square(sigma_trend)[None],
            jnp.full((4,), jnp.square(sigma_seas), dtype=dtype),
            jnp.full((regressor_count,), 1e-10, dtype=dtype),
            jnp.square(sigma_ar)[None],
        )
    )
    prior_variance = jnp.concatenate(
        (
            jnp.full((state_count - 1,), 10.0, dtype=dtype),
            (jnp.square(sigma_ar) / (1.0 - jnp.square(rho)))[None],
        )
    )
    process_covariance = jnp.diag(process_variance)
    initial_covariance = jnp.diag(prior_variance)
    observation_vector_prefix = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=dtype)
    observation_vector_suffix = jnp.array([1.0], dtype=dtype)

    def filter_step(
        carry: tuple[jax.Array, jax.Array],
        values: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        state_mean, state_covariance = carry
        observation, measurement_sd, regressor = values
        observation_vector = jnp.concatenate(
            (observation_vector_prefix, regressor, observation_vector_suffix)
        )
        predicted_mean = transition @ state_mean
        predicted_covariance = (
            transition @ state_covariance @ transition.T + process_covariance
        )
        covariance_times_observation = predicted_covariance @ observation_vector
        innovation_variance = (
            observation_vector @ covariance_times_observation
            + jnp.square(measurement_sd)
        )
        innovation = observation - observation_vector @ predicted_mean
        gain = covariance_times_observation / innovation_variance
        filtered_mean = predicted_mean + gain * innovation
        filtered_covariance = predicted_covariance - jnp.outer(
            gain, observation_vector
        ) @ predicted_covariance
        log_likelihood = -0.5 * (
            jnp.log(2.0 * jnp.pi)
            + jnp.log(innovation_variance)
            + jnp.square(innovation) / innovation_variance
        )
        return (filtered_mean, filtered_covariance), log_likelihood

    initial_mean = jnp.zeros((state_count,), dtype=dtype)
    (_, _), contributions = jax.lax.scan(
        filter_step,
        (initial_mean, initial_covariance),
        (normalized_series, normalized_uncertainty, regressors),
        length=time_count,
    )
    return jnp.sum(contributions)


def numpyro_model(
    series: jax.Array,
    uncertainty: jax.Array,
    regressors: jax.Array,
) -> None:
    sigma_trend = numpyro.sample(
        "sigma_trend", dist.HalfNormal(PRIOR_SCALES["sigma_trend"])
    )
    sigma_seas = numpyro.sample(
        "sigma_seas", dist.HalfNormal(PRIOR_SCALES["sigma_seas"])
    )
    sigma_ar = numpyro.sample("sigma_AR", dist.HalfNormal(PRIOR_SCALES["sigma_AR"]))
    rho = numpyro.sample("rhoAR1", dist.Uniform(0.0, 1.0))
    parameters = jnp.stack((sigma_trend, sigma_seas, sigma_ar, rho))
    numpyro.factor(
        "kalman_log_likelihood",
        kalman_log_likelihood(parameters, series, uncertainty, regressors),
    )


def compare_fixed_likelihood(reference: Any) -> bool:
    series = jnp.asarray(reference["series"], dtype=jnp.float64)
    uncertainty = jnp.asarray(reference["uncertainty"], dtype=jnp.float64)
    regressors = jnp.asarray(reference["regressors"], dtype=jnp.float64)
    parameters = jnp.asarray(reference["fixed_parameters"], dtype=jnp.float64)
    started = time.perf_counter()
    compiled = jax.jit(jax.value_and_grad(kalman_log_likelihood))
    log_likelihood, gradient = compiled(parameters, series, uncertainty, regressors)
    log_likelihood.block_until_ready()
    gradient.block_until_ready()
    compile_and_evaluate_seconds = time.perf_counter() - started

    reference_log_likelihood = float(reference["fixed_log_likelihood"])
    reference_gradient = np.asarray(reference["fixed_likelihood_gradient"])
    actual_log_likelihood = float(log_likelihood)
    actual_gradient = np.asarray(gradient)
    log_likelihood_error = abs(actual_log_likelihood - reference_log_likelihood)
    gradient_absolute_error = np.abs(actual_gradient - reference_gradient)
    gradient_relative_error = gradient_absolute_error / np.maximum(
        np.abs(reference_gradient), 1.0
    )
    print(f"fixed_jax_log_likelihood={actual_log_likelihood:.12g}")
    print(f"fixed_cmdstan_log_likelihood={reference_log_likelihood:.12g}")
    print(f"fixed_log_likelihood_absolute_error={log_likelihood_error:.6g}")
    print(f"fixed_jax_gradient={actual_gradient.tolist()}")
    print(f"fixed_cmdstan_gradient={reference_gradient.tolist()}")
    print(f"fixed_gradient_max_absolute_error={gradient_absolute_error.max():.6g}")
    print(f"fixed_gradient_max_relative_error={gradient_relative_error.max():.6g}")
    print(f"fixed_jax_compile_and_evaluate_seconds={compile_and_evaluate_seconds:.3f}")

    likelihood_tolerance = 1e-5 * max(1.0, abs(reference_log_likelihood))
    gradient_tolerance = 1e-4
    passed = (
        log_likelihood_error <= likelihood_tolerance
        and float(gradient_relative_error.max()) <= gradient_tolerance
    )
    print(f"fixed_parameter_parity={'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-backend-eval-real-cell-cmdstan-numpyro.npz"),
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--chain-method",
        choices=("sequential", "parallel"),
        default="sequential",
        help="Use parallel only when JAX exposes at least one CPU device per chain.",
    )
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dlmmc-backend-eval-real-cell-numpyro.npz"),
    )
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    if args.chains < 1 or args.warmup < 0 or args.samples < 4:
        parser.error(
            "chains must be positive, warmup non-negative, and samples at least 4 "
            "for NumPyro split-R-hat diagnostics"
        )
    if not args.reference.exists():
        parser.error(f"CmdStan reference not found: {args.reference}")

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    print(f"jax_version={jax.__version__} numpyro_version={numpyro.__version__}")
    print(f"jax_devices={jax.devices()}")
    if args.chain_method == "parallel" and jax.local_device_count() < args.chains:
        parser.error(
            "parallel chains require at least one JAX device per chain; "
            "configure host devices with XLA_FLAGS if appropriate"
        )

    reference = np.load(args.reference)
    parity_passed = compare_fixed_likelihood(reference)
    if args.parity_only or not parity_passed:
        if not parity_passed and not args.parity_only:
            raise SystemExit("Fixed-parameter parity failed; refusing to run NUTS.")
        return

    series = jnp.asarray(reference["series"], dtype=jnp.float64)
    uncertainty = jnp.asarray(reference["uncertainty"], dtype=jnp.float64)
    regressors = jnp.asarray(reference["regressors"], dtype=jnp.float64)
    kernel = NUTS(numpyro_model)
    sampler = MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method=args.chain_method,
        progress_bar=False,
    )

    warmup_started = time.perf_counter()
    sampler.warmup(
        jax.random.PRNGKey(args.seed),
        series,
        uncertainty,
        regressors,
    )
    warmup_seconds = time.perf_counter() - warmup_started

    sampling_started = time.perf_counter()
    sampler.run(
        jax.random.PRNGKey(args.seed + 1),
        series,
        uncertainty,
        regressors,
        extra_fields=("accept_prob", "num_steps"),
    )
    jax.tree.map(lambda value: value.block_until_ready(), sampler.get_samples())
    sampling_seconds = time.perf_counter() - sampling_started

    samples = sampler.get_samples(group_by_chain=True)
    diagnostics = numpyro_summary(samples, group_by_chain=True)
    extra_fields = sampler.get_extra_fields(group_by_chain=True)
    divergences = int(np.asarray(extra_fields["diverging"]).sum())
    acceptance = float(np.asarray(extra_fields["accept_prob"]).mean())
    total_seconds = warmup_seconds + sampling_seconds
    print(f"numpyro_warmup_and_jit_seconds={warmup_seconds:.3f}")
    print(f"numpyro_sampling_seconds={sampling_seconds:.3f}")
    print(f"numpyro_total_seconds={total_seconds:.3f}")
    print(f"numpyro_chain_method={args.chain_method}")
    print(f"numpyro_divergences={divergences}")
    print(f"numpyro_acceptance_probability_mean={acceptance:.6g}")
    print(f"numpyro_extra_fields={sorted(extra_fields)}")

    posterior_output: dict[str, np.ndarray] = {}
    for name in PARAMETER_NAMES:
        draws = np.asarray(samples[name])
        posterior_output[name] = draws
        result = diagnostics[name]
        cmdstan_draws = np.asarray(reference[name])
        print(
            f"{name}: numpyro_mean={float(draws.mean()):.8g} "
            f"cmdstan_mean={float(cmdstan_draws.mean()):.8g} "
            f"mean_delta={float(draws.mean() - cmdstan_draws.mean()):.4g} "
            f"numpyro_rhat={float(result['r_hat']):.5g} "
            f"numpyro_ess={float(result['n_eff']):.5g} "
            f"numpyro_ess_per_sampling_second={float(result['n_eff']) / sampling_seconds:.5g}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **posterior_output,
        warmup_seconds=warmup_seconds,
        sampling_seconds=sampling_seconds,
        chain_method=args.chain_method,
        divergences=divergences,
        acceptance_probability_mean=acceptance,
        fixed_log_likelihood=float(reference["fixed_log_likelihood"]),
        fixed_likelihood_gradient=np.asarray(reference["fixed_likelihood_gradient"]),
    )
    print(f"saved_posterior={args.output}")


if __name__ == "__main__":
    main()
