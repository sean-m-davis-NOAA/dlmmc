"""Compare emcee sampling with a Numba-parallel DLM log-probability callback.

This is an experimental, gradient-free sampler; it does not replace CmdStanPy.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import emcee
import numpy as np
import numba
from numba import njit, prange

from numba_real_cell import kalman_log_likelihood


PRIOR_SCALES = np.array([1e-4, 0.01, 0.5], dtype=np.float64)


@njit(parallel=True)
def log_probability_batch(positions, series, uncertainty, regressors):
    walker_count = positions.shape[0]
    log_probabilities = np.empty(walker_count, dtype=np.float64)
    for walker in prange(walker_count):
        sigma_trend, sigma_seas, sigma_ar, rho = positions[walker]
        if (
            sigma_trend <= 0.0
            or sigma_seas <= 0.0
            or sigma_ar <= 0.0
            or rho <= 0.0
            or rho >= 1.0
        ):
            log_probabilities[walker] = -np.inf
            continue
        likelihood = kalman_log_likelihood(
            positions[walker], series, uncertainty, regressors
        )
        log_prior = -0.5 * (
            (sigma_trend / PRIOR_SCALES[0]) ** 2
            + (sigma_seas / PRIOR_SCALES[1]) ** 2
            + (sigma_ar / PRIOR_SCALES[2]) ** 2
        )
        log_probabilities[walker] = likelihood + log_prior
    return log_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-backend-eval-real-cell-cmdstan-smoother.npz"),
    )
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dlmmc-numba-emcee-real-cell.npz"),
    )
    args = parser.parse_args()
    if args.walkers < 8:
        parser.error("--walkers must be at least twice the four-dimensional parameter count")
    if args.threads < 1 or args.threads > numba.config.NUMBA_NUM_THREADS:
        parser.error(f"--threads must be between 1 and NUMBA_NUM_THREADS={numba.config.NUMBA_NUM_THREADS}")
    if args.warmup < 0 or args.samples < 4:
        parser.error("warmup must be non-negative and samples must be at least 4")
    if not args.reference.exists():
        parser.error(f"CmdStan reference not found: {args.reference}")

    reference = np.load(args.reference)
    series = np.asarray(reference["series"], dtype=np.float64)
    uncertainty = np.asarray(reference["uncertainty"], dtype=np.float64)
    regressors = np.asarray(reference["regressors"], dtype=np.float64)
    cmdstan_parameters = np.array(
        [
            np.asarray(reference["sigma_trend"]).mean(),
            np.asarray(reference["sigma_seas"]).mean(),
            np.asarray(reference["sigma_AR"]).mean(),
            np.asarray(reference["rhoAR1"]).mean(),
        ],
        dtype=np.float64,
    )

    numba.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    initial_positions = np.column_stack(
        (
            np.abs(rng.normal(0.0, PRIOR_SCALES[0], args.walkers)),
            np.abs(rng.normal(0.0, PRIOR_SCALES[1], args.walkers)),
            np.abs(rng.normal(0.0, PRIOR_SCALES[2], args.walkers)),
            rng.uniform(0.05, 0.95, args.walkers),
        )
    )

    compile_started = time.perf_counter()
    initial_log_probabilities = log_probability_batch(
        initial_positions, series, uncertainty, regressors
    )
    jit_and_initial_call_seconds = time.perf_counter() - compile_started
    print(f"numba_version={numba.__version__} emcee_version={emcee.__version__}")
    print(f"numba_threads={numba.get_num_threads()} walkers={args.walkers}")
    print(f"numba_jit_and_initial_batch_seconds={jit_and_initial_call_seconds:.3f}")

    sampler = emcee.EnsembleSampler(
        args.walkers,
        4,
        log_probability_batch,
        args=(series, uncertainty, regressors),
        vectorize=True,
        moves=emcee.moves.StretchMove(a=2.0),
    )
    sampler.random_state = np.random.RandomState(args.seed).get_state()

    warmup_started = time.perf_counter()
    state = sampler.run_mcmc(initial_positions, args.warmup, progress=False)
    warmup_seconds = time.perf_counter() - warmup_started
    sampler.reset()

    sampling_started = time.perf_counter()
    sampler.run_mcmc(state, args.samples, progress=False)
    sampling_seconds = time.perf_counter() - sampling_started
    chain = sampler.get_chain()
    flat_samples = sampler.get_chain(flat=True)
    acceptance = float(np.mean(sampler.acceptance_fraction))
    try:
        autocorrelation_time = sampler.get_autocorr_time(tol=0)
        effective_samples = args.walkers * args.samples / autocorrelation_time
    except Exception as exc:
        autocorrelation_time = np.full(4, np.nan)
        effective_samples = np.full(4, np.nan)
        print(f"autocorrelation_estimate_error={type(exc).__name__}: {exc}")

    print(f"emcee_warmup_seconds={warmup_seconds:.3f}")
    print(f"emcee_sampling_seconds={sampling_seconds:.3f}")
    warmup_and_sampling_seconds = warmup_seconds + sampling_seconds
    print(f"emcee_warmup_and_sampling_seconds={warmup_and_sampling_seconds:.3f}")
    print(
        "emcee_total_including_jit_seconds="
        f"{jit_and_initial_call_seconds + warmup_and_sampling_seconds:.3f}"
    )
    print(f"emcee_acceptance_fraction_mean={acceptance:.6g}")
    print(f"emcee_autocorrelation_time={np.asarray(autocorrelation_time).tolist()}")
    print(f"emcee_estimated_ess={np.asarray(effective_samples).tolist()}")
    parameter_names = ("sigma_trend", "sigma_seas", "sigma_AR", "rhoAR1")
    for parameter_index, name in enumerate(parameter_names):
        print(
            f"{name}: emcee_mean={flat_samples[:, parameter_index].mean():.8g} "
            f"cmdstan_mean={cmdstan_parameters[parameter_index]:.8g} "
            f"mean_delta={flat_samples[:, parameter_index].mean() - cmdstan_parameters[parameter_index]:.4g}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        chain=chain,
        acceptance_fraction=sampler.acceptance_fraction,
        autocorrelation_time=autocorrelation_time,
        effective_samples=effective_samples,
        warmup_seconds=warmup_seconds,
        sampling_seconds=sampling_seconds,
        jit_and_initial_call_seconds=jit_and_initial_call_seconds,
        total_including_jit_seconds=jit_and_initial_call_seconds + warmup_and_sampling_seconds,
    )
    print(f"saved_emcee_chain={args.output}")


if __name__ == "__main__":
    main()
