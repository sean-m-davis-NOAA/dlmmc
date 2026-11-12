#!/usr/bin/env python3
"""Run the example pressure/latitude grid with the dlmmc package API."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

import dlmmc
from dlmmc.mpi import format_grid_estimate


DATASET = Path('data/BASIC_V1_2017_lotus_seascyc_gcsw2017_fac2.nc')
RESULT = Path('results/BASIC_V1_2017.nc')

# Remove entries to omit variables; use 'single' (float32) or 'double' (float64).
OUTPUT_VARIABLES: dict[str, Literal['single', 'double']] = {
    'trend': 'single',
    'sigma_trend': 'single',
    'sigma_AR': 'single',
    'rhoAR1': 'single',
    'sigma_seas': 'single',
    'beta': 'single',
}


def _interpolate_regressor(path: str, column: int, time: xr.DataArray) -> xr.DataArray:
    values = np.loadtxt(path, usecols=(0, column))
    regressor_time = pd.to_datetime(values[:, 0], unit='D', origin='julian')
    regressor = xr.DataArray(
        values[:, 1],
        dims=('time',),
        coords={'time': regressor_time},
    )
    return regressor.interp(time=time)


def _load_regressors(time: xr.DataArray, latitudes: np.ndarray) -> xr.Dataset:
    regressors = {
        'ENSO': _interpolate_regressor('regressors/ENSO_MEI_1950_201802.txt', 1, time),
        'SOLAR': _interpolate_regressor('regressors/Flux_F30_monthly_195111_201803_absolute.txt', 1, time),
        'QBO30': _interpolate_regressor('regressors/multi_qbo30_1953_2018.txt', 1, time),
        'QBO50': _interpolate_regressor('regressors/multi_qbo50_1953_2018.txt', 1, time),
    }

    saod = np.loadtxt(
        'regressors/sad_1979_2017_10deg_60S_60N.txt',
        usecols=tuple(range(len(latitudes) + 1)),
    )
    saod_time = pd.to_datetime(saod[:, 0], unit='D', origin='julian')
    saod_data = xr.DataArray(
        saod[:, 1:],
        dims=('time', 'latitude'),
        coords={'time': saod_time, 'latitude': latitudes},
    )
    regressors['SAOD'] = saod_data.interp(time=time)
    return xr.Dataset(regressors).assign_coords(time=time)


def _select_cells(data: xr.DataArray, cell_indices: list[int]) -> xr.DataArray:
    grid_dims = ('pressure', 'latitude')
    grid_shape = tuple(data.sizes[dim] for dim in grid_dims)
    cell_count = int(np.prod(grid_shape))
    if len(set(cell_indices)) != len(cell_indices):
        raise ValueError('--cell-indices must not contain duplicates.')
    if any(index < 0 or index >= cell_count for index in cell_indices):
        raise ValueError(f'--cell-indices must be between 0 and {cell_count - 1}.')

    selected = np.zeros(cell_count, dtype=bool)
    selected[cell_indices] = True
    selected_mask = xr.DataArray(
        selected.reshape(grid_shape),
        dims=grid_dims,
        coords={dim: data.coords[dim] for dim in grid_dims},
    )
    return data.where(selected_mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULT)
    parser.add_argument('--chains', type=int, default=1)
    parser.add_argument(
        '--parallel-chains',
        type=int,
        help='Maximum chains sampled in parallel for each cell.',
    )
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    cells = parser.add_mutually_exclusive_group()
    cells.add_argument(
        '--cell-indices',
        type=int,
        nargs='+',
        metavar='FLAT_INDEX',
        help='Fit only these row-major pressure/latitude indices; other cells are marked skipped.',
    )
    cells.add_argument(
        '--max-cells',
        type=int,
        metavar='N',
        help='Fit at most N cells: the first N (row-major order) not yet complete. '
        'With --resume, repeated runs continue where the previous one stopped.',
    )
    parser.add_argument('--cache-dir', type=Path, help='Shared CmdStan cache directory for multi-node runs.')
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable CmdStanPy progress bars (reduces Python overhead per chain).',
    )
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument('--overwrite', action='store_true', help='Replace an existing output file.')
    restart.add_argument(
        '--resume',
        action='store_true',
        help='Continue an existing output file, fitting only cells not yet complete '
        '(starts a new file if none exists). Use the same settings as the original run.',
    )
    dry_run = parser.add_argument_group(
        'dry run', 'Preview the cell assignment, output size, and run time without fitting or starting MPI.'
    )
    dry_run.add_argument('-n', '--dry-run', action='store_true', help='Report the plan and exit without fitting.')
    dry_run.add_argument(
        '--ranks', type=int, default=int(os.environ.get('SLURM_NTASKS', 0)) or None,
        help='Total MPI ranks, including the rank-0 writer (default: $SLURM_NTASKS).',
    )
    dry_run.add_argument(
        '--seconds-per-cell', type=float,
        help='Measured wall seconds per cell for this model, sampling settings, and node layout.',
    )
    dry_run.add_argument(
        '--safety-factor', type=float, default=1.25,
        help='Multiplier applied to the time estimate (default: 1.25).',
    )
    dry_run.add_argument('--time-limit', help='Requested Slurm --time (for example 4:00:00) to check.')
    args = parser.parse_args()
    if args.max_cells is not None and args.max_cells < 1:
        parser.error('--max-cells must be at least 1.')
    if args.dry_run and args.ranks is None:
        parser.error('--dry-run requires --ranks (or run inside a Slurm job that sets SLURM_NTASKS).')

    logging.basicConfig(format='%(asctime)s %(name)s %(message)s')
    logging.getLogger('dlmmc.mpi').setLevel(logging.INFO)

    with xr.open_dataset(DATASET) as source:
        data = source['o3']
        uncertainty = source['o3_sigma']
        decoded_time = pd.to_datetime(data['time'].values, unit='D', origin='julian')
        data = data.assign_coords(time=decoded_time)
        uncertainty = uncertainty.assign_coords(time=decoded_time)
        regressors = _load_regressors(data['time'], data['latitude'].to_numpy())
        if args.cell_indices is not None:
            try:
                data = _select_cells(data, args.cell_indices)
            except ValueError as exc:
                parser.error(str(exc))

        if args.dry_run:
            estimate = dlmmc.estimate_grid_run(
                data,
                args.ranks,
                regressors=regressors,
                model='vanilla_ar1',
                grid_dims=('pressure', 'latitude'),
                output_variables=OUTPUT_VARIABLES,
                chains=args.chains,
                samples=args.samples,
                seconds_per_cell=args.seconds_per_cell,
                safety_factor=args.safety_factor,
                time_limit=args.time_limit,
                resume_from=args.output if args.resume else None,
                max_cells=args.max_cells,
            )
            print(format_grid_estimate(estimate))
            return
        result = dlmmc.fit_grid_mpi(
            data,
            output=args.output,
            uncertainty=uncertainty,
            regressors=regressors,
            model='vanilla_ar1',
            grid_dims=('pressure', 'latitude'),
            output_variables=OUTPUT_VARIABLES,
            chains=args.chains,
            parallel_chains=args.parallel_chains,
            warmup=args.warmup,
            samples=args.samples,
            seed=args.seed,
            cache_dir=args.cache_dir,
            show_progress=not args.no_progress,
            overwrite=args.overwrite,
            resume=args.resume,
            max_cells=args.max_cells,
        )

    if result is not None:
        statuses, counts = np.unique(result['fit_status'].values, return_counts=True)
        print(f'Wrote {args.output}; fit statuses: {dict(zip(statuses.tolist(), counts.tolist()))}')
        result.close()


if __name__ == '__main__':
    main()
