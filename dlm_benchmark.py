#!/usr/bin/env python3
"""Benchmark the public dlmmc.fit API on one example ozone time series."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

import dlmmc


def load_example(pressure: int, latitude: int) -> tuple[xr.DataArray, xr.DataArray, xr.Dataset]:
    with xr.open_dataset('data/BASIC_V1_2017_lotus_seascyc_gcsw2017_fac2.nc') as source:
        data = source['o3'].isel(pressure=pressure, latitude=latitude).load()
        uncertainty = source['o3_sigma'].isel(pressure=pressure, latitude=latitude).load()

    regressors = {}
    regressor_specs = [
        ('ENSO', 'regressors/ENSO_MEI_1950_201802.txt', 1),
        ('SOLAR', 'regressors/Flux_F30_monthly_195111_201803_absolute.txt', 1),
        ('QBO30', 'regressors/multi_qbo30_1953_2018.txt', 1),
        ('QBO50', 'regressors/multi_qbo50_1953_2018.txt', 1),
        ('SAOD', 'regressors/sad_1979_2017_10deg_60S_60N.txt', latitude + 1),
    ]
    for name, path, column in regressor_specs:
        values = np.loadtxt(path, usecols=(0, column))
        regressor = xr.DataArray(
            values[:, 1],
            dims=('time',),
            coords={'time': values[:, 0]},
        )
        regressors[name] = regressor.interp(time=data['time']).to_numpy()
    regressor_dataset = xr.Dataset(
        {name: ('time', values) for name, values in regressors.items()},
        coords={'time': data['time']},
    )
    return data, uncertainty, regressor_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pressure', type=int, default=22, help='pressure index in the example dataset')
    parser.add_argument('--latitude', type=int, default=8, help='latitude index in the example dataset')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()

    data, uncertainty, regressors = load_example(args.pressure, args.latitude)
    start = time.perf_counter()
    compiled = dlmmc.compile_models(['vanilla_ar1'])
    compile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    posterior = dlmmc.fit(
        data,
        regressors=regressors,
        uncertainty=uncertainty,
        model='vanilla_ar1',
        chains=args.chains,
        warmup=args.warmup,
        samples=args.samples,
        seed=args.seed,
    )
    fit_seconds = time.perf_counter() - start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f'benchmark_{int(time.time())}.json'
    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model': 'vanilla_ar1',
        'settings': {
            'chains': args.chains,
            'warmup': args.warmup,
            'samples_per_chain': args.samples,
            'seed': args.seed,
        },
        'compile_seconds': compile_seconds,
        'fit_seconds': fit_seconds,
        'executable': compiled['vanilla_ar1'],
        'posterior_dimensions': dict(posterior.sizes),
        'posterior_variables': list(posterior.data_vars),
        'divergent_transitions': posterior.attrs.get('divergent_transitions'),
    }
    output_path.write_text(json.dumps(result, indent=2))
    print(f'Benchmark passed; results written to {output_path}')


if __name__ == '__main__':
    main()
