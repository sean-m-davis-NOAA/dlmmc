"""Create fixed-parameter CmdStan references for representative real ozone cells."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from cmdstanpy import CmdStanModel

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)

import dlmmc
from dlm_benchmark import load_example


PARAMETERS = {
    "sigma_trend": 7.732e-5,
    "sigma_seas": 0.001501,
    "sigma_AR": 0.01757,
    "rhoAR1": 0.2448,
}
PRIOR_SCALES = {
    "sigma_trend": 1e-4,
    "sigma_seas": 0.01,
    "sigma_AR": 0.5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("/tmp/dlmmc-tfp-real-stan-reference.npz"))
    args = parser.parse_args()

    grid_size = 27 * 12
    candidate_indices = np.linspace(0, grid_size - 1, args.cells + 16, dtype=int)
    cell_indices: list[int] = []
    series_rows: list[np.ndarray] = []
    stddev_rows: list[np.ndarray] = []
    regressor_rows: list[np.ndarray] = []
    regressor_names: list[str] | None = None

    for flat_index in candidate_indices:
        pressure_index, latitude_index = divmod(int(flat_index), 12)
        data, uncertainty, regressors = load_example(pressure_index, latitude_index)
        raw_series = np.asarray(data.to_numpy(), dtype=np.float64)
        raw_uncertainty = np.asarray(uncertainty.to_numpy(), dtype=np.float64)
        valid_data = np.isfinite(raw_series)
        if not valid_data.any():
            continue
        data_mean = float(raw_series[valid_data].mean())
        filled_series = np.where(valid_data, raw_series, data_mean)
        data_range = float(np.ptp(filled_series))
        if not np.isfinite(data_range) or data_range <= 0:
            continue
        missing = ~valid_data | ~np.isfinite(raw_uncertainty)
        stddev = np.where(missing, 1e20, raw_uncertainty)
        if np.any(stddev[~missing] < 0):
            raise ValueError(f"Negative uncertainty in cell {flat_index}")

        names = list(regressors.data_vars)
        regressor_values = np.column_stack(
            [np.asarray(regressors[name].to_numpy(), dtype=np.float64) for name in names]
        )
        if not np.all(np.isfinite(regressor_values)):
            raise ValueError(f"Non-finite aligned regressor in cell {flat_index}")
        if regressor_names is None:
            regressor_names = names
        elif names != regressor_names:
            raise ValueError("Regressor ordering differs between cells")

        cell_indices.append(int(flat_index))
        series_rows.append(filled_series)
        stddev_rows.append(stddev)
        regressor_rows.append(regressor_values)
        if len(cell_indices) == args.cells:
            break

    if len(cell_indices) != args.cells:
        raise ValueError(f"Found only {len(cell_indices)} valid cells; requested {args.cells}")

    series = np.stack(series_rows)
    stddev = np.stack(stddev_rows)
    regressors = np.stack(regressor_rows)
    cell_count, time_count = series.shape
    prior_parameter_term = sum(
        -0.5 * (PARAMETERS[name] / scale) ** 2
        for name, scale in PRIOR_SCALES.items()
    )
    prior_gradient = np.array(
        [
            -PARAMETERS["sigma_trend"] / PRIOR_SCALES["sigma_trend"] ** 2,
            -PARAMETERS["sigma_seas"] / PRIOR_SCALES["sigma_seas"] ** 2,
            -PARAMETERS["sigma_AR"] / PRIOR_SCALES["sigma_AR"] ** 2,
            0.0,
        ],
        dtype=np.float64,
    )

    executable = dlmmc.compile_models(["vanilla_ar1"])["vanilla_ar1"]
    stan_model = CmdStanModel(exe_file=executable)
    reference_log_likelihood = np.empty(cell_count, dtype=np.float64)
    reference_gradient = np.empty((cell_count, 4), dtype=np.float64)
    started = time.perf_counter()

    for row in range(cell_count):
        data = {
            "N": time_count,
            "nreg": regressors.shape[2],
            "sampling": 1.0,
            "time_series": series[row],
            "stddev": stddev[row],
            "regressors": regressors[row],
            "sigma_trend_prior": PRIOR_SCALES["sigma_trend"],
            "sigma_seas_prior": PRIOR_SCALES["sigma_seas"],
            "sigma_AR_prior": PRIOR_SCALES["sigma_AR"],
            "S": 10.0,
        }
        result = stan_model.log_prob(
            PARAMETERS, data, jacobian=False, sig_figs=16
        ).iloc[0]
        # CmdStan's diagnostic lp__ omits parameter-independent prior constants.
        reference_log_likelihood[row] = float(result["lp__"]) - prior_parameter_term
        unconstrained_gradient = np.array(
            [
                result["g_sigma_trend"],
                result["g_sigma_seas"],
                result["g_sigma_AR"],
                result["g_rhoAR1"],
            ],
            dtype=np.float64,
        )
        transform_derivative = np.array(
            [
                PARAMETERS["sigma_trend"],
                PARAMETERS["sigma_seas"],
                PARAMETERS["sigma_AR"],
                PARAMETERS["rhoAR1"] * (1.0 - PARAMETERS["rhoAR1"]),
            ],
            dtype=np.float64,
        )
        reference_gradient[row] = (
            unconstrained_gradient / transform_derivative - prior_gradient
        )

    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        series=series,
        stddev=stddev,
        regressors=regressors,
        parameters=np.array(
            [
                PARAMETERS["sigma_trend"],
                PARAMETERS["sigma_seas"],
                PARAMETERS["sigma_AR"],
                PARAMETERS["rhoAR1"],
            ]
        ),
        log_likelihood=reference_log_likelihood,
        gradient=reference_gradient,
        cell_indices=np.asarray(cell_indices, dtype=int),
        regressor_names=np.asarray(regressor_names),
        stan_seconds=elapsed,
    )
    print(f"cells={cell_count} time_count={time_count} regressors={regressor_names}")
    print(f"cell_indices={cell_indices}")
    print(f"stan_log_prob_total_seconds={elapsed:.6f}")
    print(f"reference_log_likelihood_sum={reference_log_likelihood.sum():.12g}")
    print(f"reference_gradient_sum={reference_gradient.sum(axis=0).tolist()}")
    print(f"saved_reference={args.output}")


if __name__ == "__main__":
    main()
