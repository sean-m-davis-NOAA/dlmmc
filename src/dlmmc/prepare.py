"""Labeled, explicit-mask preparation shared by independent production fits."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import xarray as xr
import torch
from .api import _resolve_model, _resolve_priors, _prepare_regressors, _sampling_interval
from .transforms import parameter_names
from .validation_scope import require_laptop_scope


@dataclass
class PreparedData:
    model: str
    y: torch.Tensor
    variance: torch.Tensor
    observed: torch.Tensor
    regressors: torch.Tensor
    offset: np.ndarray
    scale: np.ndarray
    time: np.ndarray
    cell_coordinates: dict
    regressor_names: list
    priors: dict
    sampling: float
    checksum: str
    input_policy: dict | None = None

    @property
    def model_spec(self):
        from .components import as_model
        return as_model(self.model)

    @property
    def state_layout(self):
        from .components import compiled_layout
        return compiled_layout(self.model,self.regressors.shape[-1])

    @property
    def cells(self): return self.y.shape[0]

    @property
    def parameter_names(self): return parameter_names(self.model,self.regressors.shape[-1])

    @property
    def arguments(self): return self.y,self.variance,self.observed,self.regressors


def prepare_data(data, regressors=None, *, uncertainty=None, model='vanilla_ar1',
                 dtype='float64', device='cpu', priors=None, sampling='monthly'):
    if not isinstance(data,xr.DataArray) or 'time' not in data.dims:
        raise ValueError('Data must be a DataArray with a time dimension')
    name,spec=_resolve_model(model)
    if dtype not in ('float32','float64'):raise ValueError('Explicit float32 or float64 required')
    device=torch.device(device)
    if device.type=='mps' and dtype=='float64':raise ValueError('MPS float64 is unsupported; no fallback')
    if device.type not in ('cpu','cuda','mps'):raise ValueError('Unsupported device')
    spatial=tuple(d for d in data.dims if d!='time')
    cells=int(np.prod([data.sizes[d] for d in spatial])) if spatial else 1
    require_laptop_scope(list(range(cells)))  # before tensor allocation or fitting
    time=np.asarray(data.time.values)
    if len(time)<2 or not np.all(time[1:]>time[:-1]):raise ValueError('Strictly increasing unique time points required')
    dt=np.diff(time.astype('datetime64[M]').astype(int)) if np.issubdtype(time.dtype,np.datetime64) else np.diff(time.astype(float))
    if not np.allclose(dt,dt[0],rtol=1e-6,atol=1e-10):raise ValueError('Irregular observation grids require a separately specified transition model')
    sampling_value=_sampling_interval(sampling)
    if not np.isfinite(sampling_value) or sampling_value<=0:raise ValueError('Positive finite sampling interval required')
    if np.issubdtype(time.dtype,np.datetime64) and not np.allclose(dt,sampling_value):
        raise ValueError('Datetime month spacing must match the declared sampling interval')
    stacked=data.stack(_cell=spatial).transpose('_cell','time') if spatial else data.expand_dims(_cell=[0]).transpose('_cell','time')
    raw=np.asarray(stacked.values,dtype=np.float64)
    if uncertainty is None:unc=np.full_like(raw,1e-20)
    elif isinstance(uncertainty,xr.DataArray):
        _,uncertainty=xr.align(data,uncertainty,join='exact')
        uncertainty=uncertainty.broadcast_like(data)
        unc=np.asarray((uncertainty.stack(_cell=spatial).transpose('_cell','time') if spatial else uncertainty.expand_dims(_cell=[0]).transpose('_cell','time')).values,dtype=float)
    else:
        array=np.asarray(uncertainty,dtype=float)
        if array.shape!=data.shape:raise ValueError('Uncertainty must match data shape')
        unc=xr.DataArray(array,dims=data.dims,coords=data.coords)
        unc=np.asarray((unc.stack(_cell=spatial).transpose('_cell','time') if spatial else unc.expand_dims(_cell=[0]).transpose('_cell','time')).values)
    finite=np.isfinite(raw)
    if np.any(~finite.any(1)):raise ValueError('All-missing cells lack a defined normalization')
    observed=finite&np.isfinite(unc)
    if np.any(unc[observed]<0):raise ValueError('Negative observation uncertainty')
    offset=np.array([x[m].mean() for x,m in zip(raw,finite)])
    scale=np.array([np.ptp(x[m]) for x,m in zip(raw,finite)])
    if np.any(scale<=0):raise ValueError('Constant series lacks a defined range normalization; no floor')
    # Sanitize before arithmetic; retain the original finite-data normalization.
    y=(np.where(observed,raw,offset[:,None])-offset[:,None])/scale[:,None]
    variance=(np.where(observed,unc,0.)/scale[:,None])**2
    if isinstance(regressors,xr.DataArray) and spatial and all(d in regressors.dims for d in spatial):
        if 'regressor' not in regressors.dims:raise ValueError('Cell-specific regressors need a regressor dimension')
        aligned=regressors.interp(time=data.time)
        for dim in spatial:
            if not np.array_equal(aligned[dim],data[dim]):raise ValueError('Regressor cell coordinates must match exactly')
        X=np.asarray(aligned.stack(_cell=spatial).transpose('_cell','time','regressor').values,dtype=float)
        names=list(map(str,aligned.regressor.values))
        if spec['regressors']=='none' or not np.isfinite(X).all() or X.shape[-1]==0:raise ValueError('Invalid model/regressor combination')
    else:
        values,names=_prepare_regressors(regressors,data.time,spec['regressors'])
        X=np.empty((cells,len(time),0)) if values is None else np.broadcast_to(values,(cells,*values.shape)).copy()
    from .components import bound_model
    name=bound_model(name,names)
    name,spec=_resolve_model(name)
    if not parameter_names(name,X.shape[-1]):
        raise ValueError('Model has no sampled parameters; use build_state_space for fixed-parameter calculations')
    prior=_resolve_priors(spec,X.shape[-1],priors,{})
    coordinates={dim:np.asarray(stacked[dim].values) for dim in spatial}
    # Include auxiliary per-panel physical coordinates (e.g. pressure, latitude).
    for key,value in stacked.coords.items():
        if value.dims==('_cell',) and key not in coordinates and key!='_cell':coordinates[key]=np.asarray(value.values)
        elif value.ndim==0 and key not in coordinates and key not in ('time','_cell'):
            coordinates[key]=np.repeat(np.asarray(value.values).reshape(1),cells)
    metadata={'model':name,'dtype':dtype,'device':str(device),'sampling':sampling_value,
              'priors':{k:np.asarray(v).tolist() for k,v in prior.items()},'regressors':names,
              'coordinates':{k:v.astype(str).tolist() for k,v in coordinates.items()}}
    digest=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode())
    for array in (raw,unc,X,time):digest.update(np.ascontiguousarray(array).tobytes())
    tdtype=getattr(torch,dtype)
    tensors=[torch.as_tensor(a,dtype=torch.bool if a.dtype==bool else tdtype,device=device) for a in (y,variance,observed,X)]
    if any(not bool(torch.isfinite(t).all()) for t in (tensors[0],tensors[1],tensors[3])):
        raise ValueError('Input representation overflows requested dtype')
    policy=None
    if not spec['seasonal'] and data.attrs.get('seasonality'):
        from .input_contract import validate_no_seasonal_input
        policy=validate_no_seasonal_input(data,name)
    return PreparedData(name,*tensors,offset,scale,time,coordinates,names,prior,sampling_value,digest.hexdigest(),policy)
