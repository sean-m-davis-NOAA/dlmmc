#!/usr/bin/env python3
"""Summarize fit status and divergences in a fit_grid_mpi() NetCDF file.

Exits with status 1 if any cell failed (fit_status -1).
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

STATUS_LABELS = {1: 'fitted', 0: 'skipped', -1: 'failed'}


def summarize(path: Path) -> tuple[list[str], bool]:
    lines = []
    with xr.open_dataset(path) as result:
        grid_dims = result['fit_status'].dims
        status = result['fit_status'].values.ravel()
        divergences = (
            result['divergent_transitions'].transpose(*grid_dims).values.ravel()
            if 'divergent_transitions' in result
            else None
        )
        draws = result.sizes.get('chain', 0) * result.sizes.get('draw', 0)

    counts = {STATUS_LABELS.get(int(value), str(value)): int((status == value).sum()) for value in np.unique(status)}
    lines.append(f'Cells: {status.size} ({", ".join(f"{count} {label}" for label, count in counts.items())})')
    failed = np.flatnonzero(status == -1)
    if failed.size:
        lines.append(f'Failed flat cell IDs: {failed.tolist()}')

    if divergences is None:
        lines.append('Divergent transitions: not stored in this file')
    else:
        fitted = status == 1
        cell_divergences = divergences[fitted]
        with_divergences = np.flatnonzero(fitted & (divergences > 0))
        lines.append(
            f'Divergent transitions: {int(cell_divergences.sum())} total, '
            f'{with_divergences.size} fitted cells with any'
            + (f' (max {int(cell_divergences.max())} of {draws} draws)' if cell_divergences.size and draws else '')
        )
        if with_divergences.size:
            worst = with_divergences[np.argsort(divergences[with_divergences])[::-1][:10]]
            lines.append(
                'Most divergences (flat ID: count): '
                + ', '.join(f'{index}: {int(divergences[index])}' for index in worst)
            )
    return lines, bool(failed.size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='NetCDF file written by fit_grid_mpi().')
    args = parser.parse_args()
    lines, failed = summarize(args.output)
    print('\n'.join(lines))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
