"""Compare a PyTorch Kalman likelihood/gradient with a CmdStan reference.

This is a fixed-parameter kernel test, not a posterior sampler or full fit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def kalman_log_likelihood(
    parameters: torch.Tensor,
    series: torch.Tensor,
    uncertainty: torch.Tensor,
    regressors: torch.Tensor,
) -> torch.Tensor:
    sigma_trend, sigma_seas, sigma_ar, rho = parameters.unbind()
    valid = torch.isfinite(series)
    series_mean = torch.where(valid, series, torch.zeros_like(series)).sum() / valid.sum()
    filled_series = torch.where(valid, series, series_mean)
    missing = (~valid) | (~torch.isfinite(uncertainty))
    filled_uncertainty = torch.where(missing, torch.full_like(uncertainty, 1e20), uncertainty)
    data_mean = filled_series.mean()
    data_range = filled_series.max() - filled_series.min()
    normalized_series = (filled_series - data_mean) / data_range
    normalized_uncertainty = filled_uncertainty / data_range

    time_count, regressor_count = regressors.shape
    state_count = regressor_count + 7
    dtype = parameters.dtype
    device = parameters.device

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
    transition = transition.clone()
    transition[-1, -1] = rho

    process_variance = torch.cat(
        (
            torch.tensor([1e-10], dtype=dtype, device=device),
            sigma_trend.square().reshape(1),
            sigma_seas.square().expand(4),
            torch.full((regressor_count,), 1e-10, dtype=dtype, device=device),
            sigma_ar.square().reshape(1),
        )
    )
    prior_variance = torch.cat(
        (
            torch.full((state_count - 1,), 10.0, dtype=dtype, device=device),
            (sigma_ar.square() / (1.0 - rho.square())).reshape(1),
        )
    )
    process_covariance = torch.diag(process_variance)
    state_mean = torch.zeros(state_count, dtype=dtype, device=device)
    state_covariance = torch.diag(prior_variance)
    observation_prefix = torch.tensor(
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=dtype, device=device
    )
    observation_suffix = torch.ones(1, dtype=dtype, device=device)
    log_likelihood = torch.zeros((), dtype=dtype, device=device)
    two_pi = torch.tensor(2.0 * np.pi, dtype=dtype, device=device)

    for time_index in range(time_count):
        observation_vector = torch.cat(
            (observation_prefix, regressors[time_index], observation_suffix)
        )
        predicted_mean = transition @ state_mean
        predicted_covariance = (
            transition @ state_covariance @ transition.T + process_covariance
        )
        covariance_times_observation = predicted_covariance @ observation_vector
        innovation_variance = (
            observation_vector @ covariance_times_observation
            + normalized_uncertainty[time_index].square()
        )
        innovation = normalized_series[time_index] - observation_vector @ predicted_mean
        gain = covariance_times_observation / innovation_variance
        state_mean = predicted_mean + gain * innovation
        state_covariance = predicted_covariance - torch.outer(
            gain, observation_vector
        ) @ predicted_covariance
        log_likelihood = log_likelihood - 0.5 * (
            torch.log(two_pi)
            + torch.log(innovation_variance)
            + innovation.square() / innovation_variance
        )
    return log_likelihood


def evaluate(device_name: str, reference: np.lib.npyio.NpzFile, repeats: int) -> None:
    device = torch.device(device_name)
    dtype = torch.float32 if device.type == "mps" else torch.float64
    series = torch.tensor(reference["series"], dtype=dtype, device=device)
    uncertainty = torch.tensor(reference["uncertainty"], dtype=dtype, device=device)
    regressors = torch.tensor(reference["regressors"], dtype=dtype, device=device)
    parameters = torch.tensor(
        reference["fixed_parameters"], dtype=dtype, device=device, requires_grad=True
    )

    started = time.perf_counter()
    likelihood = kalman_log_likelihood(parameters, series, uncertainty, regressors)
    gradient = torch.autograd.grad(likelihood, parameters)[0]
    if device.type == "mps":
        torch.mps.synchronize()
    first_call_seconds = time.perf_counter() - started

    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        likelihood = kalman_log_likelihood(parameters, series, uncertainty, regressors)
        gradient = torch.autograd.grad(likelihood, parameters)[0]
        if device.type == "mps":
            torch.mps.synchronize()
        timings.append(time.perf_counter() - started)

    likelihood_value = float(likelihood.detach().cpu())
    gradient_value = gradient.detach().cpu().numpy()
    reference_likelihood = float(reference["fixed_log_likelihood"])
    reference_gradient = np.asarray(reference["fixed_likelihood_gradient"])
    likelihood_error = abs(likelihood_value - reference_likelihood)
    gradient_relative_error = np.abs(gradient_value - reference_gradient) / np.maximum(
        np.abs(reference_gradient), 1.0
    )
    print(
        f"device={device} dtype={dtype} first_call_seconds={first_call_seconds:.4f} "
        f"warm_median_seconds={float(np.median(timings)):.6g}"
    )
    print(f"torch_log_likelihood={likelihood_value:.12g}")
    print(f"cmdstan_log_likelihood={reference_likelihood:.12g}")
    print(f"log_likelihood_absolute_error={likelihood_error:.6g}")
    print(f"torch_gradient={gradient_value.tolist()}")
    print(f"cmdstan_gradient={reference_gradient.tolist()}")
    print(f"gradient_max_relative_error={gradient_relative_error.max():.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-backend-eval-real-cell-cmdstan-smoother.npz"),
    )
    parser.add_argument("--device", choices=("cpu", "mps", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not args.reference.exists():
        parser.error(f"CmdStan reference not found: {args.reference}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.device in ("mps", "both") and not torch.backends.mps.is_available():
        parser.error("PyTorch MPS is not available in this environment")

    print(f"torch_version={torch.__version__}")
    reference = np.load(args.reference)
    devices = ("cpu", "mps") if args.device == "both" else (args.device,)
    for device_name in devices:
        evaluate(device_name, reference, args.repeats)


if __name__ == "__main__":
    main()
