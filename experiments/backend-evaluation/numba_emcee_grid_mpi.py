"""Run independent emcee/Numba DLM cell fits across local MPI workers."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import emcee
import numpy as np
import numba
from numba import njit, prange

from numba_real_cell import kalman_log_likelihood


PRIOR_SCALES = np.array([1e-4, 0.01, 0.5], dtype=np.float64)
PARAMETER_NAMES = ("sigma_trend", "sigma_seas", "sigma_AR", "rhoAR1")


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


def fit_cell(
    cell_index: int,
    series: np.ndarray,
    uncertainty: np.ndarray,
    regressors: np.ndarray,
    *,
    walkers: int,
    threads: int,
    warmup: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    numba.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    initial_positions = np.column_stack(
        (
            np.abs(rng.normal(0.0, PRIOR_SCALES[0], walkers)),
            np.abs(rng.normal(0.0, PRIOR_SCALES[1], walkers)),
            np.abs(rng.normal(0.0, PRIOR_SCALES[2], walkers)),
            rng.uniform(0.05, 0.95, walkers),
        )
    )
    started = time.perf_counter()
    log_probability_batch(initial_positions, series, uncertainty, regressors)
    jit_seconds = time.perf_counter() - started

    sampler = emcee.EnsembleSampler(
        walkers,
        4,
        log_probability_batch,
        args=(series, uncertainty, regressors),
        vectorize=True,
        moves=emcee.moves.StretchMove(a=2.0),
    )
    sampler.random_state = np.random.RandomState(seed).get_state()
    warmup_started = time.perf_counter()
    state = sampler.run_mcmc(initial_positions, warmup, progress=False)
    warmup_seconds = time.perf_counter() - warmup_started
    sampler.reset()

    sampling_started = time.perf_counter()
    sampler.run_mcmc(state, samples, progress=False)
    sampling_seconds = time.perf_counter() - sampling_started
    chain = sampler.get_chain()
    acceptance = float(np.mean(sampler.acceptance_fraction))
    try:
        tau = sampler.get_autocorr_time(tol=0)
        ess = walkers * samples / tau
    except Exception:
        tau = np.full(4, np.nan)
        ess = np.full(4, np.nan)
    return {
        "cell_index": cell_index,
        "chain": chain,
        "acceptance_fraction": np.asarray(sampler.acceptance_fraction),
        "autocorrelation_time": tau,
        "effective_samples": ess,
        "jit_seconds": jit_seconds,
        "warmup_seconds": warmup_seconds,
        "sampling_seconds": sampling_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/dlmmc-torch-real-grid-cmdstan.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/numba-emcee-grid-mpi.npz"),
    )
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if args.walkers < 8:
        parser.error("--walkers must be at least 8 for a four-parameter ensemble")
    if args.threads_per_worker < 1:
        parser.error("--threads-per-worker must be positive")
    if args.warmup < 0 or args.samples < 4:
        parser.error("warmup must be non-negative and samples must be at least 4")
    if not args.reference.exists():
        parser.error(f"fixed CmdStan reference not found: {args.reference}")

    from mpi4py import MPI

    fixed_reference = np.load(args.reference)
    cell_count = fixed_reference["series"].shape[0]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    process_count = comm.Get_size()
    if process_count > 1 and process_count - 1 > cell_count:
        parser.error("MPI fitting workers cannot exceed the number of selected cells")
    if rank == 0:
        started = time.perf_counter()
    comm.Barrier()

    if process_count == 1:
        local_cell_indices = np.arange(cell_count)
    elif rank == 0:
        local_cell_indices = np.empty(0, dtype=int)
    else:
        local_cell_indices = np.arange(cell_count)[
            np.arange(cell_count) % (process_count - 1) == rank - 1
        ]

    local_results = [
        fit_cell(
            int(cell_index),
            fixed_reference["series"][cell_index],
            fixed_reference["stddev"][cell_index],
            fixed_reference["regressors"][cell_index],
            walkers=args.walkers,
            threads=args.threads_per_worker,
            warmup=args.warmup,
            samples=args.samples,
            seed=args.seed + int(fixed_reference["cell_indices"][cell_index]),
        )
        for cell_index in local_cell_indices
    ]
    gathered = comm.gather(local_results, root=0)
    comm.Barrier()

    if rank == 0:
        by_cell = sorted(
            (result for worker_results in gathered for result in worker_results),
            key=lambda result: result["cell_index"],
        )
        if len(by_cell) != cell_count:
            raise RuntimeError(f"received {len(by_cell)} results for {cell_count} cells")
        total_seconds = time.perf_counter() - started
        chains = np.stack([result["chain"] for result in by_cell])
        acceptance = np.stack([result["acceptance_fraction"] for result in by_cell])
        autocorrelation = np.stack([result["autocorrelation_time"] for result in by_cell])
        effective_samples = np.stack([result["effective_samples"] for result in by_cell])
        jit_seconds = np.asarray([result["jit_seconds"] for result in by_cell])
        warmup_seconds = np.asarray([result["warmup_seconds"] for result in by_cell])
        sampling_seconds = np.asarray([result["sampling_seconds"] for result in by_cell])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            cell_indices=fixed_reference["cell_indices"],
            parameter_names=np.asarray(PARAMETER_NAMES),
            chain=chains,
            acceptance_fraction=acceptance,
            autocorrelation_time=autocorrelation,
            effective_samples=effective_samples,
            jit_seconds=jit_seconds,
            warmup_seconds=warmup_seconds,
            sampling_seconds=sampling_seconds,
            total_seconds=total_seconds,
            worker_count=max(1, process_count - 1),
            threads_per_worker=args.threads_per_worker,
        )
        print(f"numba_emcee_cells={cell_count} workers={max(1, process_count - 1)}")
        print(f"numba_emcee_total_wall_seconds={total_seconds:.3f}")
        print(f"numba_emcee_max_cell_sampling_seconds={sampling_seconds.max():.3f}")
        print(f"numba_emcee_acceptance_mean={acceptance.mean():.6g}")
        print(f"numba_emcee_ess_min_by_parameter={np.nanmin(effective_samples, axis=0).tolist()}")
        print(f"saved_numba_emcee_grid={args.output}")


if __name__ == "__main__":
    main()
