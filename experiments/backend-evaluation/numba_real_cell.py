"""Benchmark a Numba CPU Kalman likelihood against a fixed CmdStan reference.

This measures the forward likelihood and finite-difference gradients only; it is
not an NUTS sampler or an end-to-end DLM fit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from numba import njit


@njit
def kalman_log_likelihood(parameters, series, uncertainty, regressors):
    sigma_trend, sigma_seas, sigma_ar, rho = parameters
    time_count, regressor_count = regressors.shape
    state_count = regressor_count + 7

    data_mean = 0.0
    valid_count = 0
    for time_index in range(time_count):
        if np.isfinite(series[time_index]):
            data_mean += series[time_index]
            valid_count += 1
    data_mean /= valid_count

    filled_series = np.empty(time_count, dtype=np.float64)
    data_min = np.inf
    data_max = -np.inf
    for time_index in range(time_count):
        if np.isfinite(series[time_index]):
            value = series[time_index]
        else:
            value = data_mean
        filled_series[time_index] = value
        data_min = min(data_min, value)
        data_max = max(data_max, value)
    data_range = data_max - data_min

    normalized_series = (filled_series - data_mean) / data_range
    normalized_uncertainty = np.empty(time_count, dtype=np.float64)
    for time_index in range(time_count):
        if not np.isfinite(series[time_index]) or not np.isfinite(uncertainty[time_index]):
            value = 1e20
        else:
            value = uncertainty[time_index]
        normalized_uncertainty[time_index] = value / data_range

    transition = np.zeros((state_count, state_count), dtype=np.float64)
    transition[0, 0] = 1.0
    transition[0, 1] = 1.0
    transition[1, 1] = 1.0
    angle = 2.0 * np.pi / 12.0
    transition[2, 2] = np.cos(angle)
    transition[2, 3] = np.sin(angle)
    transition[3, 2] = -np.sin(angle)
    transition[3, 3] = np.cos(angle)
    angle = 2.0 * np.pi / 6.0
    transition[4, 4] = np.cos(angle)
    transition[4, 5] = np.sin(angle)
    transition[5, 4] = -np.sin(angle)
    transition[5, 5] = np.cos(angle)
    for regressor_index in range(regressor_count):
        transition[6 + regressor_index, 6 + regressor_index] = 1.0
    transition[state_count - 1, state_count - 1] = rho

    process_variance = np.empty(state_count, dtype=np.float64)
    process_variance[0] = 1e-10
    process_variance[1] = sigma_trend * sigma_trend
    for state_index in range(2, 6):
        process_variance[state_index] = sigma_seas * sigma_seas
    for regressor_index in range(regressor_count):
        process_variance[6 + regressor_index] = 1e-10
    process_variance[state_count - 1] = sigma_ar * sigma_ar

    filtered_mean = np.zeros(state_count, dtype=np.float64)
    filtered_covariance = np.zeros((state_count, state_count), dtype=np.float64)
    for state_index in range(state_count - 1):
        filtered_covariance[state_index, state_index] = 10.0
    filtered_covariance[state_count - 1, state_count - 1] = (
        sigma_ar * sigma_ar / (1.0 - rho * rho)
    )

    log_likelihood = 0.0
    observation_vector = np.zeros(state_count, dtype=np.float64)
    predicted_mean = np.empty(state_count, dtype=np.float64)
    left_product = np.empty((state_count, state_count), dtype=np.float64)
    predicted_covariance = np.empty((state_count, state_count), dtype=np.float64)
    observation_covariance_row = np.empty(state_count, dtype=np.float64)
    gain = np.empty(state_count, dtype=np.float64)

    for time_index in range(time_count):
        observation_vector[:] = 0.0
        observation_vector[0] = 1.0
        observation_vector[2] = 1.0
        observation_vector[4] = 1.0
        for regressor_index in range(regressor_count):
            observation_vector[6 + regressor_index] = regressors[time_index, regressor_index]
        observation_vector[state_count - 1] = 1.0

        for row in range(state_count):
            predicted_mean[row] = 0.0
            for column in range(state_count):
                predicted_mean[row] += transition[row, column] * filtered_mean[column]

        for row in range(state_count):
            for column in range(state_count):
                value = 0.0
                for inner in range(state_count):
                    value += transition[row, inner] * filtered_covariance[inner, column]
                left_product[row, column] = value
        for row in range(state_count):
            for column in range(state_count):
                value = 0.0
                for inner in range(state_count):
                    value += left_product[row, inner] * transition[column, inner]
                predicted_covariance[row, column] = value
            predicted_covariance[row, row] += process_variance[row]

        innovation_variance = normalized_uncertainty[time_index] ** 2
        innovation = normalized_series[time_index]
        for row in range(state_count):
            value = 0.0
            for column in range(state_count):
                value += predicted_covariance[row, column] * observation_vector[column]
            gain[row] = value
            innovation_variance += observation_vector[row] * value
            innovation -= observation_vector[row] * predicted_mean[row]

        for column in range(state_count):
            value = 0.0
            for row in range(state_count):
                value += observation_vector[row] * predicted_covariance[row, column]
            observation_covariance_row[column] = value

        gain /= innovation_variance
        for row in range(state_count):
            filtered_mean[row] = predicted_mean[row] + gain[row] * innovation
        for row in range(state_count):
            for column in range(state_count):
                filtered_covariance[row, column] = (
                    predicted_covariance[row, column]
                    - gain[row] * observation_covariance_row[column]
                )

        log_likelihood += -0.5 * (
            np.log(2.0 * np.pi)
            + np.log(innovation_variance)
            + innovation * innovation / innovation_variance
        )
    return log_likelihood


def finite_difference_gradient(parameters, series, uncertainty, regressors):
    gradient = np.empty_like(parameters)
    for parameter_index, parameter in enumerate(parameters):
        step = max(abs(parameter) * 1e-5, 1e-10)
        plus = parameters.copy()
        minus = parameters.copy()
        plus[parameter_index] += step
        minus[parameter_index] -= step
        gradient[parameter_index] = (
            kalman_log_likelihood(plus, series, uncertainty, regressors)
            - kalman_log_likelihood(minus, series, uncertainty, regressors)
        ) / (2.0 * step)
    return gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-backend-eval-real-cell-cmdstan-smoother.npz"),
    )
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.reference.exists():
        parser.error(f"CmdStan reference not found: {args.reference}")

    reference = np.load(args.reference)
    parameters = np.asarray(reference["fixed_parameters"], dtype=np.float64)
    series = np.asarray(reference["series"], dtype=np.float64)
    uncertainty = np.asarray(reference["uncertainty"], dtype=np.float64)
    regressors = np.asarray(reference["regressors"], dtype=np.float64)

    started = time.perf_counter()
    likelihood = kalman_log_likelihood(parameters, series, uncertainty, regressors)
    cold_seconds = time.perf_counter() - started

    timings = np.empty(args.repeats, dtype=np.float64)
    for repeat in range(args.repeats):
        started = time.perf_counter()
        likelihood = kalman_log_likelihood(parameters, series, uncertainty, regressors)
        timings[repeat] = time.perf_counter() - started
    gradient = finite_difference_gradient(parameters, series, uncertainty, regressors)
    reference_likelihood = float(reference["fixed_log_likelihood"])
    reference_gradient = np.asarray(reference["fixed_likelihood_gradient"])
    gradient_relative_error = np.abs(gradient - reference_gradient) / np.maximum(
        np.abs(reference_gradient), 1.0
    )

    print(f"numba_version={__import__('numba').__version__}")
    print(f"numba_cold_compile_and_call_seconds={cold_seconds:.6f}")
    print(f"numba_warm_median_seconds={float(np.median(timings)):.6g}")
    print(f"numba_warm_calls_per_second={1.0 / float(np.median(timings)):.6g}")
    print(f"numba_log_likelihood={likelihood:.12g}")
    print(f"cmdstan_log_likelihood={reference_likelihood:.12g}")
    print(f"log_likelihood_absolute_error={abs(likelihood - reference_likelihood):.6g}")
    print(f"numba_finite_difference_gradient={gradient.tolist()}")
    print(f"cmdstan_gradient={reference_gradient.tolist()}")
    print(f"gradient_max_relative_error={gradient_relative_error.max():.6g}")
    print("note=forward likelihood benchmark only; finite differences are not a NUTS gradient backend")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            likelihood=likelihood,
            gradient=gradient,
            cold_seconds=cold_seconds,
            warm_seconds=timings,
        )
        print(f"saved_result={args.output}")


if __name__ == "__main__":
    main()
