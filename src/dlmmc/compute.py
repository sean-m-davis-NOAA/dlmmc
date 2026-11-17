"""Discover compute at sampler initialization; explicit choices never fall back."""
from dataclasses import dataclass
from contextlib import contextmanager
import os

@dataclass(frozen=True)
class ComputePolicy:
    backend:str
    device:str
    dtype:str
    validation:str


def available_compute():
    """Cheap device discovery, without allocating tensors or compiling kernels.

    CmdStan-only installations do not need Torch just to report CPU execution.
    """
    from importlib.util import find_spec
    result = {'cuda': False, 'mps': False, 'numba': find_spec('numba') is not None,
              'torch': False}
    try:
        import torch
    except ImportError:
        return result
    result.update(cuda=bool(torch.cuda.is_available()),
                  mps=bool(torch.backends.mps.is_available()), torch=True)
    return result


def _available_text(available):
    labels = {'cuda':'CUDA', 'mps':'MPS', 'numba':'Numba CPU', 'torch':'Torch CPU'}
    return ', '.join(labels[k] for k in labels if available[k]) or 'CPU (CmdStan)'


def select_backend(backend=None, *, device=None, dtype=None, announce=False):
    """Choose CUDA, then MPS, then Numba; respect explicit device/dtype choices."""
    automatic = backend is None or backend == 'auto'
    available = available_compute() if automatic or announce else None
    if automatic:
        if device is not None:
            kind = str(device).split(':')[0]
            backend = 'numba' if kind == 'cpu' else kind
        elif dtype is not None and str(dtype).removeprefix('torch.') == 'float64':
            backend = 'numba'
        else:
            backend = next((b for b in ('cuda','mps') if available[b]), 'numba')
    if backend == 'cpu':backend = 'numba'
    if backend not in ('numba','torch','mps','cuda'):
        raise ValueError('Backend must be auto, numba/cpu, torch, mps or cuda')
    if device is not None and str(device).split(':')[0] != ('cpu' if backend in ('numba','torch') else backend):
        raise ValueError('Backend and device disagree; no device fallback')
    if announce:
        chosen_dtype = str(dtype).removeprefix('torch.') if dtype is not None else ('float32' if backend in ('cuda','mps') else 'float64')
        chosen_device = str(device) if device is not None else ('cpu' if backend in ('numba','torch') else backend)
        print(f"DLMMC Affine: available compute: {_available_text(available)}. "
              f"Selected backend='{backend}', device='{chosen_device}', dtype='{chosen_dtype}'. "
              "Change with backend='cuda', 'mps', 'numba' (CPU), or 'torch' (CPU); "
              "backend='auto' selects CUDA > MPS > Numba CPU.", flush=True)
    return backend


def announce_cmdstan():
    print(f"DLMMC CmdStan: available compute: {_available_text(available_compute())}. "
          "Selected CPU (float64); CmdStan NUTS uses CPU. To use a GPU, choose "
          "sampler='affine', backend='auto' (or backend='cuda'/'mps').", flush=True)


def resolve_compute(backend=None,*,device=None,dtype=None,check_available=True,announce=False):
    import torch
    backend=select_backend(backend,device=device,dtype=dtype)
    wanted='cpu' if backend in ('numba','torch') else backend
    dev=torch.device(wanted if device is None else device)
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
    if announce:select_backend(backend,device=dev,dtype=automatic,announce=True)
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
    if backend not in ('mps','cuda'):
        if prepared.y.device.type == 'cpu':return prepared
        from dataclasses import replace
        return replace(prepared, **{name:getattr(prepared,name).cpu().contiguous()
            for name in ('y','variance','observed','regressors')})
    policy=resolve_compute(backend)
    from dataclasses import replace
    import torch
    return replace(prepared,**{name:getattr(prepared,name).to(device=policy.device,dtype=torch.bool if name=='observed' else torch.float32).contiguous()
        for name in ('y','variance','observed','regressors')})
