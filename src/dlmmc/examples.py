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
