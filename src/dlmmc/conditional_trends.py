"""Deterministic fixed-parameter trend moments across model/backend options.

This operation does not declare an MCMC window converged. Values are physical
mean, variance, anchor-change mean and variance, with shape [draw,cell,time,4].
"""
import numpy as np
import torch
from .sampling import make_target
from .compute import gpu_arithmetic

@torch.no_grad()
def conditional_trend_moments(prepared,parameters,anchor,*,backend='numba',regressors=None,workspace_bytes=1024**3):
    p=prepared;u=parameters
    if backend=='cpu':backend='numba'
    from .components import require_level
    require_level(p.model,p.regressors.shape[-1])
    if not isinstance(u,torch.Tensor) or u.ndim!=3 or u.shape[1:]!=(p.cells,len(p.parameter_names)) or u.dtype!=p.y.dtype or u.device!=p.y.device:
        raise ValueError('Parameters must match prepared dtype/device and model dimension')
    if not len(u) or not 0<=anchor<len(p.time):raise ValueError('Nonempty draws and valid anchor required')
    if regressors is not None and (regressors.shape!=(len(u),p.cells,len(p.time),p.regressors.shape[-1]) or regressors.dtype!=u.dtype or regressors.device!=u.device):raise ValueError('Per-draw regressors have incompatible shape/dtype/device')
    if backend=='mps':
        from .mps_smoothing import fixed_mps_trend_moments
        return fixed_mps_trend_moments(p,u,anchor,workspace_bytes=workspace_bytes,regressors=regressors)
    if backend in ('torch','cuda'):
        if backend=='cuda' and (u.device.type!='cuda' or u.dtype not in (torch.float32,torch.float64)):raise ValueError('CUDA requires native float32/64')
        from .gpu_smoothing import fixed_cuda_trend_moments
        with gpu_arithmetic(u.device):return fixed_cuda_trend_moments(p,u,anchor,regressors=regressors)
    if backend!='numba' or u.device.type!='cpu' or u.dtype not in (torch.float32,torch.float64):raise ValueError('Numba requires CPU32/64; choose a supported backend')
    if regressors is None:
        from .production_smoothing import fixed_grid_moments
        return torch.from_numpy(fixed_grid_moments(p,u.numpy(),anchor))
    from .tensor_target import model_components
    from .production_trends import _trend_batch
    F,G,Q,C,_,valid=model_components(u,p.y,p.variance,p.observed,regressors,p.model,S=p.priors['S'],sampling=p.sampling)
    if not bool(valid.all()):raise ValueError('Invalid retained draw; no removal')
    out=[]
    for g in range(p.cells):
        args=tuple(np.ascontiguousarray(t.numpy()) for t in (p.y[g],p.variance[g],p.observed[g],F[:,g],G[:,g],Q[:,g],C[:,g]))
        value=_trend_batch(*args,anchor);value[...,0]=value[...,0]*p.scale[g]+p.offset[g];value[...,1]*=p.scale[g]**2;value[...,2]*=p.scale[g];value[...,3]*=p.scale[g]**2;out.append(value)
    return torch.from_numpy(np.stack(out,axis=1))
