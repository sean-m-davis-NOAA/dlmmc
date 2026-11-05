#!/usr/bin/env python3
"""Verify that exactly the selected example-grid cells completed."""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('cell_indices', type=int, nargs='+')
    args = parser.parse_args()

    with xr.open_dataset(args.output) as result:
        status = result['fit_status'].transpose('pressure', 'latitude').values.ravel()

    selected = np.asarray(args.cell_indices)
    if len(set(args.cell_indices)) != len(args.cell_indices):
        parser.error('cell indices must not contain duplicates.')
    if np.any(selected < 0) or np.any(selected >= status.size):
        parser.error(f'cell indices must be between 0 and {status.size - 1}.')

    values, counts = np.unique(status, return_counts=True)
    print('fit_status_counts', dict(zip(values.tolist(), counts.tolist())))
    expected = np.zeros_like(status)
    expected[selected] = 1
    mismatches = np.flatnonzero(status != expected)
    if mismatches.size:
        raise SystemExit(f'Unexpected fit_status at flat cell IDs {mismatches.tolist()}.')


if __name__ == '__main__':
    main()
