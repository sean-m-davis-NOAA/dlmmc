#!/usr/bin/env python3
"""Run the example pressure/latitude grid with the dlmmc package API."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import dlmmc


DATASET = Path('data/BASIC_V1_2017_lotus_seascyc_gcsw2017_fac2.nc')
RESULT = Path('results/BASIC_V1_2017.nc')


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULT)
    parser.add_argument('--chains', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cache-dir', type=Path, help='Shared CmdStan cache directory for multi-node runs.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    with xr.open_dataset(DATASET) as source:
        data = source['o3']
        uncertainty = source['o3_sigma']
        decoded_time = pd.to_datetime(data['time'].values, unit='D', origin='julian')
        data = data.assign_coords(time=decoded_time)
        uncertainty = uncertainty.assign_coords(time=decoded_time)
        regressors = _load_regressors(data['time'], data['latitude'].to_numpy())
        result = dlmmc.fit_grid_mpi(
            data,
            output=args.output,
            uncertainty=uncertainty,
            regressors=regressors,
            model='vanilla_ar1',
            grid_dims=('pressure', 'latitude'),
            chains=args.chains,
            warmup=args.warmup,
            samples=args.samples,
            seed=args.seed,
            cache_dir=args.cache_dir,
            overwrite=args.overwrite,
        )

    if result is not None:
        statuses, counts = np.unique(result['fit_status'].values, return_counts=True)
        print(f'Wrote {args.output}; fit statuses: {dict(zip(statuses.tolist(), counts.tolist()))}')
        result.close()


if __name__ == '__main__':
    main()
