"""Benchmark Numba parallelism across independent fixed-parameter DLM cells."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numba
import numpy as np
from numba import njit, prange

from numba_real_cell import kalman_log_likelihood


@njit(parallel=True)
def evaluate_grid(parameters, series, stddev, regressors):
    cell_count = series.shape[0]
    likelihoods = np.empty(cell_count, dtype=np.float64)
    for cell_index in prange(cell_count):
        likelihoods[cell_index] = kalman_log_likelihood(
            parameters,
            series[cell_index],
            stddev[cell_index],
            regressors[cell_index],
        )
    return likelihoods


@njit(parallel=True)
def gradient_grid(parameters, series, stddev, regressors):
    cell_count = series.shape[0]
    parameter_count = parameters.shape[0]
    gradients = np.empty((cell_count, parameter_count), dtype=np.float64)
    for cell_index in prange(cell_count):
        for parameter_index in range(parameter_count):
            step = max(abs(parameters[parameter_index]) * 1e-5, 1e-10)
            plus = parameters.copy()
            minus = parameters.copy()
            plus[parameter_index] += step
            minus[parameter_index] -= step
            gradients[cell_index, parameter_index] = (
                kalman_log_likelihood(
                    plus,
                    series[cell_index],
                    stddev[cell_index],
                    regressors[cell_index],
                )
                - kalman_log_likelihood(
                    minus,
                    series[cell_index],
                    stddev[cell_index],
                    regressors[cell_index],
                )
            ) / (2.0 * step)
    return gradients


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-tfp-real-stan-reference.npz"),
    )
    parser.add_argument("--threads", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not args.reference.exists():
        parser.error(f"CmdStan grid reference not found: {args.reference}")
    if any(count < 1 or count > numba.config.NUMBA_NUM_THREADS for count in args.threads):
        parser.error(
            f"thread counts must be between 1 and NUMBA_NUM_THREADS={numba.config.NUMBA_NUM_THREADS}"
        )
    if len(set(args.threads)) != len(args.threads):
        parser.error("thread counts must be unique")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    reference = np.load(args.reference)
    parameters = np.asarray(reference["parameters"], dtype=np.float64)
    series = np.asarray(reference["series"], dtype=np.float64)
    stddev = np.asarray(reference["stddev"], dtype=np.float64)
    regressors = np.asarray(reference["regressors"], dtype=np.float64)
    expected_likelihoods = np.asarray(reference["log_likelihood"], dtype=np.float64)
    expected_gradients = np.asarray(reference["gradient"], dtype=np.float64)

    for thread_count in args.threads:
        numba.set_num_threads(thread_count)
        started = time.perf_counter()
        likelihoods = evaluate_grid(parameters, series, stddev, regressors)
        cold_likelihood_seconds = time.perf_counter() - started
        started = time.perf_counter()
        gradients = gradient_grid(parameters, series, stddev, regressors)
        cold_gradient_seconds = time.perf_counter() - started

        likelihood_timings = np.empty(args.repeats)
        gradient_timings = np.empty(args.repeats)
        for repeat in range(args.repeats):
            started = time.perf_counter()
            likelihoods = evaluate_grid(parameters, series, stddev, regressors)
            likelihood_timings[repeat] = time.perf_counter() - started
            started = time.perf_counter()
            gradients = gradient_grid(parameters, series, stddev, regressors)
            gradient_timings[repeat] = time.perf_counter() - started

        likelihood_errors = np.abs(likelihoods - expected_likelihoods)
        gradient_relative_errors = np.abs(gradients - expected_gradients) / np.maximum(
            np.abs(expected_gradients), 1.0
        )
        likelihood_seconds = float(np.median(likelihood_timings))
        gradient_seconds = float(np.median(gradient_timings))
        print(
            f"threads={thread_count} cells={len(series)} "
            f"cold_likelihood_compile_seconds={cold_likelihood_seconds:.3f} "
            f"cold_gradient_compile_seconds={cold_gradient_seconds:.3f} "
            f"warm_likelihood_seconds={likelihood_seconds:.6g} "
            f"likelihood_cells_per_second={len(series) / likelihood_seconds:.5g} "
            f"warm_finite_difference_gradient_seconds={gradient_seconds:.6g}"
        )
        print(
            f"likelihood_mean_abs_error={likelihood_errors.mean():.6g} "
            f"likelihood_max_abs_error={likelihood_errors.max():.6g} "
            f"gradient_max_relative_error={gradient_relative_errors.max():.6g}"
        )
    print("note=forward likelihood and finite-difference parity only; not a sampler benchmark")


if __name__ == "__main__":
    main()
