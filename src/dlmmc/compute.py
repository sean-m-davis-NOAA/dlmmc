"""Explicit production compute policies; never fall back to another device."""
from dataclasses import dataclass
from contextlib import contextmanager
import os

@dataclass(frozen=True)
class ComputePolicy:
    backend:str
    device:str
    dtype:str
    validation:str


def resolve_compute(backend='numba',*,device=None,dtype=None,check_available=True):
    import torch
    if backend=='cpu':backend='numba'
    if backend not in ('numba','torch','mps','cuda'):raise ValueError('Production backend must be numba/cpu, torch, mps or cuda')
    wanted='cpu' if backend in ('numba','torch') else backend
    dev=torch.device(wanted if device is None else device)
    if dev.type!=wanted:raise ValueError('Backend and device disagree; no device fallback')
    automatic='float64' if wanted=='cpu' else 'float32'
    if dtype is not None and str(dtype).removeprefix('torch.')!=automatic:
        raise ValueError(f'{backend} production requires {automatic}; choose CPU for the float64 oracle')
    if wanted=='mps' and check_available:
        for key in ('PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH'):
            if os.environ.get(key,'0')!='0':raise ValueError(f'{key} must be0 for native production arithmetic')
            os.environ.setdefault(key,'0')
    if check_available:
        if wanted=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Native MPS unavailable; no CPU fallback')
        if wanted=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; native testing pending; no CPU fallback')
    return ComputePolicy(backend,str(dev),automatic,'SCOPED_CPU64' if wanted=='cpu' else
        'MPS_USER_APPROVED_PRODUCTION_WITH_RETAINED_PRECISION_FLAGS' if wanted=='mps' else 'CUDA_UNTESTED')


@contextmanager
def gpu_arithmetic(device):
    import torch
    if torch.device(device).type!='cuda':
        yield;return
    old=torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        yield
    finally:torch.set_float32_matmul_precision(old)


def guarded_gpu_arithmetic(function):
    from functools import wraps
    @wraps(function)
    def wrapped(fit,*args,**kwargs):
        with gpu_arithmetic(fit.prepared.y.device):return function(fit,*args,**kwargs)
    return wrapped


def prepared_for_backend(prepared,backend):
    """Explicit backend selection transfers prepared inputs; no fallback."""
    if backend not in ('mps','cuda'):return prepared
    policy=resolve_compute(backend)
    from dataclasses import replace
    import torch
    return replace(prepared,**{name:getattr(prepared,name).to(device=policy.device,dtype=torch.bool if name=='observed' else torch.float32).contiguous()
        for name in ('y','variance','observed','regressors')})
