"""Load the small, provenance-checked tutorial bundle shipped in the checkout."""
from pathlib import Path
import json
import hashlib
import numpy as np
import xarray as xr


def load_panels(directory):
    folder=Path(directory);metadata=json.loads((folder/'provenance.json').read_text())
    path=folder/'panels.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['panels_sha256']:
        raise ValueError('Example bundle checksum mismatch')
    with np.load(path,allow_pickle=False) as z:
        coords={'time':z['time'],'panel':metadata['panel_ids'],'pressure':('panel',z['pressure']),
                'latitude':('panel',z['latitude'])}
        data=xr.DataArray(z['observations'],dims=('time','panel'),coords=coords,name='ozone',attrs={'units':'ppmv'})
        uncertainty=xr.DataArray(z['uncertainty'],dims=data.dims,coords=coords,name='uncertainty')
        regressors=xr.DataArray(z['regressors'],dims=('panel','time','regressor'),coords={'panel':metadata['panel_ids'],
            'time':z['time'],'regressor':metadata['regressor_names']})
    return data,uncertainty,regressors,metadata


def load_scientific_example(directory):
    """Saved science for plots, separate from fresh short-kernel smoke fits."""
    folder=Path(directory);metadata=json.loads((folder/'provenance.json').read_text())
    path=folder/'scientific_plot_subset.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['scientific_subset_sha256']:
        raise ValueError('Scientific example checksum mismatch')
    with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
    return values,metadata


def load_trend_summary(directory):
    """Read and verify production absolute trend summaries, never anchored changes."""
    folder = Path(directory)
    record = json.loads((folder/'execution.json').read_text())
    means, sds, panels, flags = [], [], [], []
    time = None
    for row in record['cells']:
        path = folder/f"cell-{row['cell']:04d}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Trend summary checksum mismatch')
        with np.load(path, allow_pickle=False) as z:
            if time is not None and not np.array_equal(time, z['time']):
                raise ValueError('Trend summary time mismatch')
            time = z['time'].copy()
            means.append(z['mean'].copy()); sds.append(z['sd'].copy())
        panels.append(row['panel']); flags.append(';'.join(row['flags']))
    if not panels:
        raise ValueError('No trend summaries found')
    return xr.Dataset({'trend': (('panel', 'time'), np.asarray(means)),
                       'trend_sd': (('panel', 'time'), np.asarray(sds)),
                       'diagnostic_flags': ('panel', flags)},
                      coords={'panel': panels, 'time': time},
                      attrs={'sampler': 'affine', 'fit_provenance': record['fit_provenance'],
                             'uncertainty': 'pointwise posterior SD; unanchored',
                             'qualification_status': record['status']})


def stratospheric_panels(dataset):
    """Paper domain: 1–100 hPa tropics, supplied deeper levels at |latitude|>30.

    Returns named observation/error panels in pressure-major input order. This
    selects data only; it grants no fitting authorization or spatial scope.
    """
    pressure = np.asarray(dataset.pressure)
    latitude = np.asarray(dataset.latitude)
    indices = [(i, j) for i, p in enumerate(pressure) for j, lat in enumerate(latitude)
               if p >= 1 and (p <= 100 or abs(lat) > 30)]
    if not indices:
        raise ValueError('No panels in the stratospheric domain')
    pi, li = zip(*indices)
    selected = dataset.isel(pressure=xr.DataArray(list(pi), dims='panel'),
                            latitude=xr.DataArray(list(li), dims='panel'))
    selected = selected.assign_coords(panel=[f'p{i:02d}-lat{j:02d}' for i,j in indices])
    return selected.o3.transpose('time', 'panel'), selected.o3_sigma.transpose('time', 'panel')


def plot_trend_grid(summary, *, output_dir=None, rows_per_page=9):
    """Pressure × latitude atlas of absolute trend ±1 posterior SD.

    No anchor subtraction or conditioning. Requires trend/trend_sd(panel,time)
    and pressure/latitude(panel) coordinates. Returns figures for notebook use.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    if type(rows_per_page) is not int or rows_per_page < 1:
        raise ValueError('Positive rows_per_page required')
    for name in ('trend', 'trend_sd'):
        if set(summary[name].dims) != {'panel', 'time'}:
            raise ValueError('Grid trends require panel/time dimensions')
    pressures = np.unique(summary.pressure.values)
    latitudes = np.unique(summary.latitude.values)
    pairs = list(zip(summary.pressure.values, summary.latitude.values))
    if len(set(pairs)) != len(pairs):
        raise ValueError('Duplicate pressure/latitude cells')
    figures = []
    for start in range(0, len(pressures), rows_per_page):
        levels = pressures[start:start+rows_per_page]
        fig, axes = plt.subplots(len(levels), len(latitudes), squeeze=False,
                                 figsize=(2.4*len(latitudes), 1.8*len(levels)))
        for i, p in enumerate(levels):
            for j, lat in enumerate(latitudes):
                ax = axes[i, j]
                indices = np.flatnonzero((summary.pressure.values == p) & (summary.latitude.values == lat))
                if not len(indices):
                    ax.set_visible(False); continue
                cell = summary.isel(panel=int(indices[0]))
                mean, sd = cell.trend.values, cell.trend_sd.values
                ax.plot(cell.time, mean, color='tab:blue', lw=1)
                ax.fill_between(cell.time, mean-sd, mean+sd, color='tab:blue', alpha=.22)
                ax.set_title(f'{p:.2f} hPa, {lat:+.0f}°', fontsize=8)
                ax.xaxis.set_major_locator(mdates.YearLocator(10))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.tick_params(labelsize=6)
                if j == 0:
                    ax.set_ylabel('Ozone (ppmv)', fontsize=7)
        fig.suptitle('Absolute ozone trend ±1 posterior SD — no anchoring', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .97)); figures.append(fig)
        if output_dir is not None:
            folder = Path(output_dir); folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder/f'trend-grid-{len(figures)}.png', dpi=120)
    return figures
