"""Benchmark batched PyTorch likelihoods for independent real DLM grid cells.

This measures the forward likelihood and gradients, not posterior sampling.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def batched_kalman_log_likelihood(parameters, series, stddev, regressors):
    sigma_trend, sigma_seas, sigma_ar, rho = parameters.unbind(dim=1)
    batch_size, _ = series.shape
    regressor_count = regressors.shape[2]
    state_count = regressor_count + 7
    dtype = parameters.dtype
    device = parameters.device

    series_mean = series.mean(dim=1, keepdim=True)
    data_range = series.amax(dim=1, keepdim=True) - series.amin(dim=1, keepdim=True)
    normalized_series = (series - series_mean) / data_range
    normalized_stddev = stddev / data_range

    transition_np = np.zeros((state_count, state_count), dtype=np.float64)
    transition_np[0, 0] = 1.0
    transition_np[0, 1] = 1.0
    transition_np[1, 1] = 1.0
    for start, angle in ((2, 2.0 * np.pi / 12.0), (4, 2.0 * np.pi / 6.0)):
        transition_np[start, start] = np.cos(angle)
        transition_np[start, start + 1] = np.sin(angle)
        transition_np[start + 1, start] = -np.sin(angle)
        transition_np[start + 1, start + 1] = np.cos(angle)
    if regressor_count:
        transition_np[6 : 6 + regressor_count, 6 : 6 + regressor_count] = np.eye(
            regressor_count, dtype=np.float64
        )
    transition = torch.tensor(transition_np, dtype=dtype, device=device)
    transition = transition.expand(batch_size, state_count, state_count).clone()
    transition[:, -1, -1] = rho

    process_variance = torch.cat(
        (
            torch.full((batch_size, 1), 1e-10, dtype=dtype, device=device),
            sigma_trend.square()[:, None],
            sigma_seas.square()[:, None].expand(batch_size, 4),
            torch.full((batch_size, regressor_count), 1e-10, dtype=dtype, device=device),
            sigma_ar.square()[:, None],
        ),
        dim=1,
    )
    prior_variance = torch.cat(
        (
            torch.full((batch_size, state_count - 1), 10.0, dtype=dtype, device=device),
            (sigma_ar.square() / (1.0 - rho.square()))[:, None],
        ),
        dim=1,
    )
    process_covariance = torch.diag_embed(process_variance)
    state_mean = torch.zeros((batch_size, state_count), dtype=dtype, device=device)
    state_covariance = torch.diag_embed(prior_variance)
    observation_prefix = torch.tensor(
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=dtype, device=device
    ).expand(batch_size, 6)
    observation_suffix = torch.ones((batch_size, 1), dtype=dtype, device=device)
    log_likelihood = torch.zeros(batch_size, dtype=dtype, device=device)
    two_pi = torch.tensor(2.0 * np.pi, dtype=dtype, device=device)

    for time_index in range(series.shape[1]):
        observation_vector = torch.cat(
            (observation_prefix, regressors[:, time_index, :], observation_suffix), dim=1
        )
        predicted_mean = torch.bmm(transition, state_mean.unsqueeze(-1)).squeeze(-1)
        predicted_covariance = (
            torch.bmm(
                torch.bmm(transition, state_covariance), transition.transpose(1, 2)
            )
            + process_covariance
        )
        covariance_times_observation = torch.bmm(
            predicted_covariance, observation_vector.unsqueeze(-1)
        ).squeeze(-1)
        innovation_variance = (
            (observation_vector * covariance_times_observation).sum(dim=1)
            + normalized_stddev[:, time_index].square()
        )
        innovation = normalized_series[:, time_index] - (
            observation_vector * predicted_mean
        ).sum(dim=1)
        gain = covariance_times_observation / innovation_variance[:, None]
        state_mean = predicted_mean + gain * innovation[:, None]
        observation_covariance_row = torch.bmm(
            observation_vector.unsqueeze(1), predicted_covariance
        )
        state_covariance = predicted_covariance - gain.unsqueeze(2) * observation_covariance_row
        log_likelihood = log_likelihood - 0.5 * (
            torch.log(two_pi)
            + torch.log(innovation_variance)
            + innovation.square() / innovation_variance
        )
    return log_likelihood


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-torch-real-grid-cmdstan.npz"),
    )
    parser.add_argument("--device", choices=("cpu", "mps", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, help="CPU intra-op threads; ignored on MPS.")
    args = parser.parse_args()
    if not args.reference.exists():
        parser.error(f"CmdStan reference not found: {args.reference}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    if args.device in ("mps", "both") and not torch.backends.mps.is_available():
        parser.error("PyTorch MPS is not available in this environment")

    reference = np.load(args.reference)
    batch_size = reference["series"].shape[0]
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device_names = ("cpu", "mps") if args.device == "both" else (args.device,)
    print(
        f"torch_version={torch.__version__} cells={batch_size} "
        f"cpu_threads={torch.get_num_threads()}"
    )
    for device_name in device_names:
        device = torch.device(device_name)
        dtype = torch.float32 if device.type == "mps" else torch.float64
        series = torch.tensor(reference["series"], dtype=dtype, device=device)
        stddev = torch.tensor(reference["stddev"], dtype=dtype, device=device)
        regressors = torch.tensor(reference["regressors"], dtype=dtype, device=device)
        parameter_values = np.broadcast_to(
            reference["parameters"], (batch_size, 4)
        ).copy()
        parameters = torch.tensor(
            parameter_values, dtype=dtype, device=device, requires_grad=True
        )

        started = time.perf_counter()
        likelihoods = batched_kalman_log_likelihood(parameters, series, stddev, regressors)
        gradients = torch.autograd.grad(likelihoods.sum(), parameters)[0]
        if device.type == "mps":
            torch.mps.synchronize()
        first_call_seconds = time.perf_counter() - started

        timings = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            likelihoods = batched_kalman_log_likelihood(parameters, series, stddev, regressors)
            gradients = torch.autograd.grad(likelihoods.sum(), parameters)[0]
            if device.type == "mps":
                torch.mps.synchronize()
            timings.append(time.perf_counter() - started)

        actual_likelihoods = likelihoods.detach().cpu().numpy()
        actual_gradients = gradients.detach().cpu().numpy()
        expected_likelihoods = np.asarray(reference["log_likelihood"])
        expected_gradients = np.asarray(reference["gradient"])
        likelihood_errors = np.abs(actual_likelihoods - expected_likelihoods)
        gradient_relative_errors = np.abs(actual_gradients - expected_gradients) / np.maximum(
            np.abs(expected_gradients), 1.0
        )
        median_seconds = float(np.median(timings))
        print(
            f"device={device} dtype={dtype} first_call_seconds={first_call_seconds:.4f} "
            f"warm_median_seconds={median_seconds:.6g} "
            f"cells_per_second={batch_size / median_seconds:.4g}"
        )
        print(
            f"likelihood_mean_abs_error={likelihood_errors.mean():.6g} "
            f"likelihood_max_abs_error={likelihood_errors.max():.6g}"
        )
        print(f"gradient_max_relative_error={gradient_relative_errors.max():.6g}")


if __name__ == "__main__":
    main()
