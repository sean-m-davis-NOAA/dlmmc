"""Compare batched TFP likelihoods/gradients on CPU and GPU to CmdStan references."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


def make_model_builder(reference: np.lib.npyio.NpzFile):
    series = reference["series"]
    stddev = reference["stddev"]
    regressors = reference["regressors"]
    parameters = reference["parameters"].astype(np.float32)
    cell_count, time_count = series.shape
    nreg = regressors.shape[2]
    state_count = nreg + 7

    series_range = np.ptp(series, axis=1, keepdims=True)
    scaled_series = (series - series.mean(axis=1, keepdims=True)) / series_range
    scaled_stddev = stddev / series_range
    observation_values = np.zeros(
        (time_count, cell_count, 1, state_count), dtype=np.float32
    )
    observation_values[:, :, 0, 0] = 1.0
    observation_values[:, :, 0, 2] = 1.0
    observation_values[:, :, 0, 4] = 1.0
    observation_values[:, :, 0, 6 : 6 + nreg] = np.transpose(regressors, (1, 0, 2))
    observation_values[:, :, 0, -1] = 1.0
    observation_variance = np.transpose(scaled_stddev**2, (1, 0))[:, :, None]
    observed = tf.constant(scaled_series[:, :, None], dtype=tf.float32)
    observation_matrix_values = tf.constant(observation_values, dtype=tf.float32)
    observation_variance_values = tf.constant(observation_variance, dtype=tf.float32)

    transition_np = np.zeros((state_count, state_count), dtype=np.float32)
    transition_np[:2, :2] = [[1.0, 1.0], [0.0, 1.0]]
    for first, angle in ((2, 2 * np.pi / 12), (4, 2 * np.pi / 6)):
        transition_np[first : first + 2, first : first + 2] = [
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)],
        ]
    transition_np[6 : 6 + nreg, 6 : 6 + nreg] = np.eye(nreg, dtype=np.float32)
    transition_base = tf.constant(transition_np, dtype=tf.float32)

    def build_batched_model(theta):
        sigma_trend, sigma_season, sigma_ar, rho = tf.unstack(theta)
        transition = tf.tensor_scatter_nd_update(
            transition_base, [[state_count - 1, state_count - 1]], [rho]
        )
        process_variance = tf.concat(
            [
                tf.constant([1e-10], dtype=tf.float32),
                tf.reshape(sigma_trend**2, [1]),
                tf.fill([4], sigma_season**2),
                tf.fill([nreg], tf.constant(1e-10, tf.float32)),
                tf.reshape(sigma_ar**2, [1]),
            ],
            axis=0,
        )
        state_prior_variance = tf.concat(
            [
                tf.fill([state_count - 1], tf.constant(10.0, tf.float32)),
                [sigma_ar**2 / (1.0 - rho**2)],
            ],
            axis=0,
        )
        initial_covariance = (
            transition
            @ tf.linalg.diag(state_prior_variance)
            @ tf.transpose(transition)
            + tf.linalg.diag(process_variance)
        )
        initial_scale = tf.linalg.cholesky(initial_covariance)
        initial_prior = tfd.MultivariateNormalTriL(
            loc=tf.zeros([cell_count, state_count], dtype=tf.float32),
            scale_tril=tf.broadcast_to(
                initial_scale[None, :, :], [cell_count, state_count, state_count]
            ),
        )
        return tfd.LinearGaussianStateSpaceModel(
            num_timesteps=time_count,
            transition_matrix=transition,
            transition_noise=tfd.MultivariateNormalDiag(
                loc=tf.zeros([state_count], dtype=tf.float32),
                scale_diag=tf.sqrt(process_variance),
            ),
            observation_matrix=lambda step: tf.linalg.LinearOperatorFullMatrix(
                tf.gather(observation_matrix_values, step, axis=0)
            ),
            observation_noise=lambda step: tfd.MultivariateNormalDiag(
                loc=tf.zeros([cell_count, 1], dtype=tf.float32),
                scale_diag=tf.sqrt(
                    tf.gather(observation_variance_values, step, axis=0)
                ),
            ),
            initial_state_prior=initial_prior,
            experimental_parallelize=False,
        )

    def evaluate(theta):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            cell_log_likelihood = build_batched_model(theta).log_prob(observed)
            total_log_likelihood = tf.reduce_sum(cell_log_likelihood)
        return cell_log_likelihood, tape.gradient(total_log_likelihood, theta)

    return evaluate, observed, parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-tfp-real-stan-reference.npz"),
    )
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    reference = np.load(args.reference)
    evaluate, observed, parameters = make_model_builder(reference)
    stan_log_likelihood = reference["log_likelihood"]
    stan_gradient = reference["gradient"]
    cell_count = observed.shape[0]
    tf.config.set_soft_device_placement(False)
    print("tensorflow", tf.__version__, "tfp", tfp.__version__)
    print("real_cells", cell_count, "time_count", observed.shape[1])
    print("stan_log_prob_diagnostic_seconds", float(reference["stan_seconds"]))

    devices = {
        "cpu": "/CPU:0",
        "gpu": "/GPU:0",
    }
    selected_devices = devices if args.device == "both" else {args.device: devices[args.device]}
    for label, device in selected_devices.items():
        with tf.device(device):
            compiled_evaluate = tf.function(evaluate)
            theta = tf.constant(parameters, dtype=tf.float32)
            first_log_likelihood, first_gradient = compiled_evaluate(theta)
            first_log_likelihood.numpy()
            first_gradient.numpy()
            timings = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                cell_log_likelihood, gradient = compiled_evaluate(theta)
                likelihood_values = cell_log_likelihood.numpy()
                gradient_values = gradient.numpy()
                timings.append(time.perf_counter() - started)

        median_seconds = float(np.median(timings))
        likelihood_error = likelihood_values - stan_log_likelihood
        gradient_error = gradient_values - stan_gradient.sum(axis=0)
        gradient_relative_error = np.abs(gradient_error) / np.maximum(
            np.abs(stan_gradient.sum(axis=0)), 1.0
        )
        print(
            "device", label,
            "log_likelihood_device", cell_log_likelihood.device,
            "gradient_device", gradient.device,
            "median_seconds", median_seconds,
            "cells_per_second", cell_count / median_seconds,
            "ll_error_mean_abs", float(np.mean(np.abs(likelihood_error))),
            "ll_error_median_abs", float(np.median(np.abs(likelihood_error))),
            "ll_error_max_abs", float(np.max(np.abs(likelihood_error))),
            "gradient_max_relative_error", float(np.max(gradient_relative_error)),
        )

    print(
        "Note: CmdStanModel.log_prob is a diagnostic subprocess API, not a fair ",
        "sampler-throughput baseline. Compare end-to-end fits separately.",
    )


if __name__ == "__main__":
    main()
