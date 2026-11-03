"""Benchmark vectorized NumPyro fits for independent real DLM grid cells."""

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

from numpyro_real_cell import (
    PARAMETER_NAMES,
    PRIOR_SCALES,
    draw_smoothed_states,
    kalman_log_likelihood,
    simulation_smoother_draw,
)

STATE_NAMES = ("trend", "slope", "seasonal", "ar", "beta")


def load_cmdstan_posterior(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".nc":
        import xarray as xr

        with xr.open_dataset(path) as dataset:
            cell_indices = np.asarray(dataset["cell"].values)
            result: dict[str, np.ndarray] = {"cell_indices": cell_indices}
            for name in PARAMETER_NAMES:
                result[f"draws_{name}"] = np.asarray(
                    dataset[name].transpose("cell", "chain", "draw").values
                )
            result["draws_beta"] = np.asarray(
                dataset["beta"].transpose("cell", "chain", "draw", "regressor").values
            )
            for name in STATE_NAMES[:-1]:
                draws = np.asarray(
                    dataset[name].transpose("cell", "chain", "draw", "time").values
                )
                result[f"mean_{name}"] = draws.mean(axis=(1, 2))
                result[f"sd_{name}"] = draws.std(axis=(1, 2))
            beta_draws = result["draws_beta"]
            result["mean_beta"] = beta_draws.mean(axis=(1, 2))
            result["sd_beta"] = beta_draws.std(axis=(1, 2))
            return result

    with np.load(path) as source:
        result = {name: np.asarray(source[name]) for name in source.files}
    if "cell_indices" not in result:
        raise ValueError(f"CmdStan posterior reference lacks cell indices: {path}")
    return result


def numpyro_grid_model(
    series: jax.Array,
    stddev: jax.Array,
    regressors: jax.Array,
) -> None:
    cell_count = series.shape[0]
    with numpyro.plate("cell", cell_count):
        sigma_trend = numpyro.sample(
            "sigma_trend", dist.HalfNormal(PRIOR_SCALES["sigma_trend"])
        )
        sigma_seas = numpyro.sample(
            "sigma_seas", dist.HalfNormal(PRIOR_SCALES["sigma_seas"])
        )
        sigma_ar = numpyro.sample(
            "sigma_AR", dist.HalfNormal(PRIOR_SCALES["sigma_AR"])
        )
        rho = numpyro.sample("rhoAR1", dist.Uniform(0.0, 1.0))
        parameters = jnp.stack((sigma_trend, sigma_seas, sigma_ar, rho), axis=-1)
        cell_log_likelihoods = jax.vmap(kalman_log_likelihood)(
            parameters, series, stddev, regressors
        )
        numpyro.factor("cell_likelihood", cell_log_likelihoods)


def compare_fixed_grid(reference: Any) -> bool:
    series = jnp.asarray(reference["series"], dtype=jnp.float64)
    stddev = jnp.asarray(reference["stddev"], dtype=jnp.float64)
    regressors = jnp.asarray(reference["regressors"], dtype=jnp.float64)
    parameters = jnp.broadcast_to(
        jnp.asarray(reference["parameters"], dtype=jnp.float64),
        (series.shape[0], 4),
    )
    started = time.perf_counter()
    evaluate_grid = jax.jit(
        jax.vmap(jax.value_and_grad(kalman_log_likelihood))
    )
    likelihoods, gradients = evaluate_grid(parameters, series, stddev, regressors)
    likelihoods.block_until_ready()
    gradients.block_until_ready()
    elapsed = time.perf_counter() - started

    likelihood_errors = np.abs(
        np.asarray(likelihoods) - np.asarray(reference["log_likelihood"])
    )
    expected_gradients = np.asarray(reference["gradient"])
    gradient_relative_errors = np.abs(np.asarray(gradients) - expected_gradients) / np.maximum(
        np.abs(expected_gradients), 1.0
    )
    print(f"fixed_grid_jax_compile_and_evaluate_seconds={elapsed:.3f}")
    print(f"fixed_grid_likelihood_mean_abs_error={likelihood_errors.mean():.6g}")
    print(f"fixed_grid_likelihood_max_abs_error={likelihood_errors.max():.6g}")
    print(f"fixed_grid_gradient_max_relative_error={gradient_relative_errors.max():.6g}")

    likelihood_tolerance = 1e-5 * max(
        1.0, float(np.max(np.abs(reference["log_likelihood"])))
    )
    gradient_tolerance = 1e-4
    passed = (
        float(likelihood_errors.max()) <= likelihood_tolerance
        and float(gradient_relative_errors.max()) <= gradient_tolerance
    )
    print(f"fixed_grid_parameter_parity={'PASS' if passed else 'FAIL'}")
    return passed


def draw_batched_smoothed_states(
    parameter_draws: np.ndarray,
    series: jax.Array,
    stddev: jax.Array,
    regressors: jax.Array,
    seed: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    draw_count, cell_count, _ = parameter_draws.shape
    all_keys = jax.random.split(jax.random.PRNGKey(seed), draw_count * cell_count)
    all_keys = all_keys.reshape((draw_count, cell_count, 2))

    def smooth_all_cells(keys, parameters):
        return jax.vmap(simulation_smoother_draw, in_axes=(0, 0, 0, 0, 0))(
            keys,
            parameters,
            series,
            stddev,
            regressors,
        )

    smooth_draw_batch = jax.jit(
        jax.vmap(smooth_all_cells, in_axes=(0, 0))
    )
    outputs: list[tuple[np.ndarray, ...]] = []
    started = time.perf_counter()
    for start in range(0, draw_count, batch_size):
        stop = min(start + batch_size, draw_count)
        batch_outputs = smooth_draw_batch(
            all_keys[start:stop],
            jnp.asarray(parameter_draws[start:stop], dtype=jnp.float64),
        )
        batch_outputs[0].block_until_ready()
        outputs.append(tuple(np.asarray(value) for value in batch_outputs))
    elapsed = time.perf_counter() - started
    stacked = [
        np.concatenate([output[index] for output in outputs], axis=0)
        for index in range(len(STATE_NAMES))
    ]
    print(f"numpyro_batched_smoother_seconds={elapsed:.3f}")
    return dict(zip(STATE_NAMES, stacked))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-torch-real-grid-cmdstan.npz"),
    )
    parser.add_argument(
        "--cmdstan-reference",
        type=Path,
        default=Path("/tmp/dlmmc-cmdstan-multicell-posterior.npz"),
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--chain-method", choices=("sequential", "parallel"), default="parallel")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--smooth-batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dlmmc-numpyro-multicell-posterior.npz"),
    )
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    if args.chains < 1 or args.warmup < 0 or args.samples < 4:
        parser.error("chains must be positive, warmup non-negative, and samples at least 4")
    if args.smooth_batch_size < 1:
        parser.error("--smooth-batch-size must be positive")
    if not args.reference.exists():
        parser.error(f"fixed CmdStan reference not found: {args.reference}")
    if not args.cmdstan_reference.exists() and not args.parity_only:
        parser.error(f"posterior CmdStan reference not found: {args.cmdstan_reference}")

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    print(f"jax_version={jax.__version__} numpyro_version={numpyro.__version__}")
    print(f"jax_devices={jax.devices()}")

    fixed_reference = np.load(args.reference)
    if not compare_fixed_grid(fixed_reference):
        raise SystemExit("Fixed-grid likelihood/gradient parity failed; refusing to sample.")
    if args.parity_only:
        return

    posterior_reference = load_cmdstan_posterior(args.cmdstan_reference)
    if not np.array_equal(
        np.asarray(fixed_reference["cell_indices"]),
        posterior_reference["cell_indices"],
    ):
        parser.error("fixed-likelihood and CmdStan posterior references use different cells")
    series = jnp.asarray(fixed_reference["series"], dtype=jnp.float64)
    stddev = jnp.asarray(fixed_reference["stddev"], dtype=jnp.float64)
    regressors = jnp.asarray(fixed_reference["regressors"], dtype=jnp.float64)
    kernel = NUTS(numpyro_grid_model)
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
        stddev,
        regressors,
    )
    warmup_seconds = time.perf_counter() - warmup_started
    sampling_started = time.perf_counter()
    sampler.run(
        jax.random.PRNGKey(args.seed + 1),
        series,
        stddev,
        regressors,
        extra_fields=("accept_prob", "num_steps"),
    )
    samples = sampler.get_samples(group_by_chain=True)
    jax.tree.map(lambda value: value.block_until_ready(), samples)
    sampling_seconds = time.perf_counter() - sampling_started

    extra_fields = sampler.get_extra_fields(group_by_chain=True)
    divergences = int(np.asarray(extra_fields["diverging"]).sum())
    acceptance = float(np.asarray(extra_fields["accept_prob"]).mean())
    print(f"numpyro_grid_warmup_jit_seconds={warmup_seconds:.3f}")
    print(f"numpyro_grid_sampling_seconds={sampling_seconds:.3f}")
    print(f"numpyro_grid_divergences={divergences}")
    print(f"numpyro_grid_acceptance_mean={acceptance:.6g}")

    for name in PARAMETER_NAMES:
        cell_rhats = []
        cell_ess = []
        for cell_index in range(series.shape[0]):
            cell_summary = numpyro_summary(
                {name: samples[name][:, :, cell_index]},
                group_by_chain=True,
            )[name]
            cell_rhats.append(float(cell_summary["r_hat"]))
            cell_ess.append(float(cell_summary["n_eff"]))
        numpyro_means = np.asarray(samples[name]).mean(axis=(0, 1))
        cmdstan_means = np.asarray(posterior_reference[f"draws_{name}"]).mean(axis=(1, 2))
        print(
            f"{name}: mean_abs_cell_delta={np.abs(numpyro_means - cmdstan_means).mean():.6g} "
            f"max_rhat={np.nanmax(cell_rhats):.5g} "
            f"min_ess={np.nanmin(cell_ess):.5g}"
        )

    flat_parameters = np.stack(
        [np.asarray(samples[name]).reshape(-1, series.shape[0]) for name in PARAMETER_NAMES],
        axis=-1,
    )
    smoother_started = time.perf_counter()
    smoothed = draw_batched_smoothed_states(
        flat_parameters,
        series,
        stddev,
        regressors,
        args.seed + 2,
        args.smooth_batch_size,
    )
    smoother_seconds = time.perf_counter() - smoother_started

    output: dict[str, np.ndarray] = {
        "cell_indices": np.asarray(fixed_reference["cell_indices"]),
        "warmup_seconds": np.asarray(warmup_seconds),
        "sampling_seconds": np.asarray(sampling_seconds),
        "smoother_seconds": np.asarray(smoother_seconds),
        "divergences": np.asarray(divergences),
        "acceptance_probability_mean": np.asarray(acceptance),
    }
    for name in PARAMETER_NAMES:
        output[f"draws_{name}"] = np.asarray(samples[name])
    for name, draws in smoothed.items():
        output[f"mean_{name}"] = draws.mean(axis=0)
        output[f"sd_{name}"] = draws.std(axis=0)
        chain_draws = draws.reshape(
            args.chains,
            args.samples,
            draws.shape[1],
            *draws.shape[2:],
        )
        if name == "beta":
            chain_draws = chain_draws.transpose(0, 1, 3, 2)
        else:
            chain_draws = chain_draws.transpose(0, 1, 3, 2)
        output[f"draws_{name}"] = chain_draws
        cmdstan_means = np.asarray(posterior_reference[f"mean_{name}"])
        cmdstan_sds = np.asarray(posterior_reference[f"sd_{name}"])
        rmse = float(np.sqrt(np.mean(np.square(draws.mean(axis=0) - cmdstan_means))))
        scale = float(np.sqrt(np.mean(np.square(cmdstan_sds))))
        print(f"{name}_posterior_mean_rmse={rmse:.6g} rmse_over_cmdstan_sd={rmse / scale:.6g}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_started = time.perf_counter()
    np.savez_compressed(args.output, **output)
    output_seconds = time.perf_counter() - output_started
    output["output_seconds"] = np.asarray(output_seconds)
    print(f"saved_numpyro_grid={args.output}")
    print(f"numpyro_output_compression_seconds={output_seconds:.3f}")
    print(
        "numpyro_total_sampling_smoothing_and_output_seconds="
        f"{warmup_seconds + sampling_seconds + smoother_seconds + output_seconds:.3f}"
    )


if __name__ == "__main__":
    main()
