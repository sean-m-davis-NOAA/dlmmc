"""Fused structured Metal filter specialized for all supported DLM families.

Explicit float32 MPS only. No precision fallback, numerical
floors or posterior qualification. One GPU thread evaluates each walker/cell;
the full time loop and both scalar-observation Joseph factors stay in the kernel.
"""
import os
import time
from dataclasses import replace
from functools import lru_cache
import torch
from .tensor_target import model_components
from .torch_filter import TorchTarget
from .components import compiled_layout, regressor_count


METAL_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
constant uint D = 12;

kernel void structured_filter(
    device const float* y, device const float* variance, device const bool* observed,
    device const float* F, device const float* G, device const float* Q,
    device const float* C, device float* output, device int* failure,
    constant uint& cells, constant uint& times, constant uint& count, constant uint& varying_F,
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    uint cell = index % cells;
    uint matrix = index * D * D;
    float P[D*D], tmp[D*D], B[D*D];
    float m[D], mp[D], f[D], p[D], gain[D], bf[D];
    uint col0[D], col1[D];
    float g0[D], g1[D];
    for (uint i=0; i<D; ++i) {
        m[i] = 0.0f;
        col0[i] = i; col1[i] = i;
        if (i == 0) col1[i] = 1;
        if (i == 2 || i == 3) { col0[i]=2; col1[i]=3; }
        if (i == 4 || i == 5) { col0[i]=4; col1[i]=5; }
        g0[i] = G[matrix+i*D+col0[i]];
        g1[i] = col0[i] == col1[i] ? 0.0f : G[matrix+i*D+col1[i]];
        for (uint j=0; j<D; ++j) P[i*D+j] = C[matrix+i*D+j];
    }
    float total=0.0f;
    failure[index] = -1;
    for (uint t=0; t<times; ++t) {
        for (uint i=0; i<D; ++i) {
            mp[i] = g0[i]*m[col0[i]] + g1[i]*m[col1[i]];
            for (uint j=0; j<D; ++j)
                tmp[i*D+j] = g0[i]*P[col0[i]*D+j] + g1[i]*P[col1[i]*D+j];
        }
        for (uint i=0; i<D; ++i) {
            m[i] = mp[i];
            for (uint j=0; j<D; ++j)
                P[i*D+j] = Q[matrix+i*D+j] + tmp[i*D+col0[j]]*g0[j] + tmp[i*D+col1[j]]*g1[j];
        }
        uint at=cell*times+t;
        if (!observed[at]) continue;
        for (uint i=0; i<D; ++i) f[i]=F[((varying_F ? index : cell)*times+t)*D+i];
        float s=variance[at], residual=y[at];
        for (uint i=0; i<D; ++i) {
            float value=0.0f;
            for (uint j=0; j<D; ++j) value += P[i*D+j]*f[j];
            p[i]=value; s+=f[i]*value; residual-=f[i]*m[i];
        }
        if (!isfinite(s) || s<=0.0f) {
            output[index]=NAN; failure[index]=int(t); return;
        }
        for (uint i=0; i<D; ++i) { gain[i]=p[i]/s; m[i]+=gain[i]*residual; }
        // B=(I-k f')P; retain nonsymmetric roundoff, then the second factor.
        for (uint j=0; j<D; ++j) {
            float value=0.0f;
            for (uint h=0; h<D; ++h) value+=f[h]*P[h*D+j];
            for (uint i=0; i<D; ++i) B[i*D+j]=P[i*D+j]-gain[i]*value;
        }
        for (uint i=0; i<D; ++i) {
            float value=0.0f;
            for (uint j=0; j<D; ++j) value+=B[i*D+j]*f[j];
            bf[i]=value;
        }
        for (uint i=0; i<D; ++i)
            for (uint j=0; j<D; ++j)
                P[i*D+j]=B[i*D+j]-bf[i]*gain[j]+gain[i]*gain[j]*variance[at];
        total+=-0.5f*(log(6.2831853071795864769f*s)+residual*residual/s);
    }
    output[index]=total;
}
'''


def state_layout(model,nreg):
    layout=compiled_layout(model,nreg)
    return layout.dimension,layout.ar_start


def transition_code(model,nreg,*,smoother=False):
    layout=compiled_layout(model,nreg)
    a,b=('c0','c1') if smoother else ('col0','col1')
    return '\n'.join(f'if (i=={i}) {{ {a}[i]={x}; {b}[i]={y}; }}'
                     for i,(x,y) in enumerate(zip(*layout.transition_columns)) if x!=i or y!=i)


def specialize_source(source,model,nreg):
    d,_=state_layout(model,nreg)
    source=source.replace('constant uint D = 12;',f'constant uint D = {d};')
    source=source.replace('if (i == 0) col1[i] = 1;', '')
    old='if (i == 2 || i == 3) { col0[i]=2; col1[i]=3; }\n        if (i == 4 || i == 5) { col0[i]=4; col1[i]=5; }'
    return source.replace(old,transition_code(model,nreg))


@lru_cache(maxsize=64)
def compile_kernel(model="vanilla_ar1", nreg=5):
    """Explicit compilation/cache boundary; callers warm their exact shapes."""
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0':
        raise ValueError('Launch with PYTORCH_ENABLE_MPS_FALLBACK=0 and PYTORCH_MPS_FAST_MATH=0')
    if not torch.backends.mps.is_available():raise RuntimeError('Native MPS device unavailable')
    before=time.perf_counter();library=torch.mps.compile_shader(specialize_source(METAL_SOURCE,model,nreg))
    kernel=library.structured_filter  # Resolve the compute pipeline before timing.
    return kernel, time.perf_counter()-before


def filter_mps(y,variance,observed,F,G,Q,C,*,model="vanilla_ar1"):
    """Full-batch GPU likelihood; F is shared per panel, not copied per walker.

    Matrices must match the selected model and regressor count from model_components;
    this low-level implementation is intentionally not a generic sparse filter.
    """
    tensors=(y,variance,F,G,Q,C)
    if any(x.requires_grad for x in tensors):raise ValueError('Metal filter has no autograd implementation')
    if any(x.device.type!='mps' or x.dtype!=torch.float32 or not x.is_contiguous() for x in tensors):
        raise ValueError('Contiguous MPS float32 tensors required; no implicit transfer/cast')
    if observed.device.type!='mps' or observed.dtype!=torch.bool or not observed.is_contiguous():
        raise ValueError('Contiguous MPS Boolean observation mask required')
    if y.ndim!=2 or y.shape!=variance.shape or y.shape!=observed.shape or G.ndim!=4:
        raise ValueError('Panel/time data and walker/panel matrices required')
    cells,times=y.shape;walkers=G.shape[0]
    d=G.shape[-1];nreg=regressor_count(model,d)
    state_layout(model,nreg)
    if not times or not walkers or F.shape not in ((cells,times,d),(walkers,cells,times,d)) or G.shape!=(walkers,cells,d,d) or Q.shape!=G.shape or C.shape!=G.shape:
        raise ValueError('Model state dimension and all matrix shapes must match')
    if max(F.numel(),walkers*cells*d*d)>=2**32:raise ValueError('Metal uint32 indexing limit exceeded')
    kernel,_=compile_kernel(model,nreg)
    out=y.new_empty((walkers,cells));failed=torch.empty((walkers,cells),dtype=torch.int32,device=y.device)
    kernel(y,variance,observed,F,G,Q,C,out,failed,cells,times,walkers*cells,int(F.ndim==4),
        threads=walkers*cells,group_size=min(64,walkers*cells))
    if bool((failed>=0).any()):raise FloatingPointError('Invalid innovation variance; no floor')
    if not bool(torch.isfinite(out).all()):raise FloatingPointError('Nonfinite structured MPS likelihood')
    return out


class StructuredMPSTarget(TorchTarget):
    """Native MPS32 target; model priors and Jacobians come from the shared specification."""
    def __init__(self,prepared):
        p=prepared
        state_layout(p.model,p.regressors.shape[-1])
        if p.y.device.type!='mps' or p.y.dtype!=torch.float32:
            raise ValueError('Explicit native MPS float32 data required')
        # Prepared xarray-backed data may be column-major. Copy layout once on
        # the same device/dtype, outside callback timing; never copy per walker.
        p=replace(p,**{name:getattr(p,name).contiguous() for name in
            ('y','variance','observed','regressors')})
        super().__init__(p)

    def __call__(self,u):
        p=self.prepared
        if u.requires_grad:raise ValueError('Experimental Metal target has no autograd implementation')
        if u.ndim!=3 or u.shape[1:]!=(p.cells,len(p.parameter_names)) or u.device.type!='mps' or u.dtype!=torch.float32:
            raise ValueError('Expected MPS float32 [walkers,panels,parameters] parameters')
        _,G,Q,C,prior,valid=model_components(u,*self.empty,p.model,S=p.priors['S'],sampling=p.sampling)
        value=filter_mps(p.y,p.variance,p.observed,self.F,G,Q,C,model=p.model)+prior+self.prior_correction(u,valid)
        if bool((valid&~torch.isfinite(value)).any()):raise FloatingPointError('Nonfinite valid target')
        return torch.where(valid,value,torch.full_like(value,-torch.inf))
