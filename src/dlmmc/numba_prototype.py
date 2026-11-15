"""Explicit float64 CPU qualification candidate; stock Torch affine callback.

The dense baseline and optional structured candidate retain both Joseph factors.
The latter uses exact transition zeros and scalar-observation algebra. Neither
uses fastmath, jitter or changed model policy. Only current-state workspace is
retained during a likelihood evaluation. Both require their own qualification.
"""
import math
import time
import numpy as np
import torch
from numba import njit,prange,typeof
from .tensor_target import model_components


@njit(cache=True,fastmath=False)
def _one(y,variance,observed,F,G,Q,C):
    n,d=F.shape;m=np.zeros(d);P=C.copy();eye=np.eye(d);total=0.
    for t in range(n):
        m=G@m;P=G@P@G.T+Q
        if observed[t]:
            f=F[t];p=P@f;s=f@p+variance[t]
            if not np.isfinite(s) or s<=0:return np.nan,t
            v=y[t]-f@m;k=p/s
            A=eye-np.outer(k,f)
            m=m+k*v
            P=A@P@A.T+np.outer(k,k)*variance[t]
            total+=-.5*(math.log(2*math.pi*s)+v*v/s)
    return total,-1


@njit(cache=True,parallel=True,fastmath=False)
def _batch(y,variance,observed,F,G,Q,C):
    count=F.shape[0];cells=y.shape[0]
    values=np.empty(count);failure=np.empty(count,dtype=np.int64)
    for i in prange(count):
        values[i],failure[i]=_one(y[i%cells],variance[i%cells],observed[i%cells],F[i],G[i],Q[i],C[i])
    return values,failure


@njit(cache=True,fastmath=False)
def _one_structured(y,variance,observed,F,G,Q,C):
    """Same Joseph update using the exact zeros of G and scalar-observation rank.

    B=(I-k f')P, then P=B(I-f k')+k v k'. No variance floors or
    subtractive P-k p' replacement of the complete Joseph expression.
    """
    n,d=F.shape;m=np.zeros(d);P=C.copy();total=0.
    columns=np.empty((d,d),dtype=np.int64);counts=np.zeros(d,dtype=np.int64)
    for i in range(d):
        for j in range(d):
            if G[i,j]!=0:
                columns[i,counts[i]]=j;counts[i]+=1
    tmp=np.empty((d,d));pred=np.empty((d,d));B=np.empty((d,d))
    mp=np.empty(d);p=np.empty(d);gain=np.empty(d);bf=np.empty(d)
    for t in range(n):
        for i in range(d):
            value=0.
            for h in range(counts[i]):value+=G[i,columns[i,h]]*m[columns[i,h]]
            mp[i]=value
            for j in range(d):
                value=0.
                for h in range(counts[i]):value+=G[i,columns[i,h]]*P[columns[i,h],j]
                tmp[i,j]=value
        for i in range(d):
            m[i]=mp[i]
            for j in range(d):
                value=Q[i,j]
                for h in range(counts[j]):value+=tmp[i,columns[j,h]]*G[j,columns[j,h]]
                pred[i,j]=value
        if not observed[t]:
            P[:,:]=pred
            continue
        s=variance[t];residual=y[t]
        for i in range(d):
            value=0.
            for j in range(d):value+=pred[i,j]*F[t,j]
            p[i]=value;s+=F[t,i]*value;residual-=F[t,i]*m[i]
        if not np.isfinite(s) or s<=0:return np.nan,t
        for i in range(d):
            gain[i]=p[i]/s;m[i]+=gain[i]*residual
        # Preserve both Joseph factors, allowing nonsymmetric roundoff in pred.
        for j in range(d):
            value=0.
            for h in range(d):value+=F[t,h]*pred[h,j]
            for i in range(d):B[i,j]=pred[i,j]-gain[i]*value
        for i in range(d):
            value=0.
            for j in range(d):value+=B[i,j]*F[t,j]
            bf[i]=value
        for i in range(d):
            for j in range(d):P[i,j]=B[i,j]-bf[i]*gain[j]+gain[i]*gain[j]*variance[t]
        total+=-.5*(math.log(2*math.pi*s)+residual*residual/s)
    return total,-1


@njit(cache=True,parallel=True,fastmath=False)
def _batch_structured(y,variance,observed,F,G,Q,C):
    count=F.shape[0];cells=y.shape[0];values=np.empty(count);failure=np.empty(count,dtype=np.int64)
    for i in prange(count):
        values[i],failure[i]=_one_structured(y[i%cells],variance[i%cells],observed[i%cells],F[i],G[i],Q[i],C[i])
    return values,failure


def compile_specialization(u,y,variance,observed,X,model,*,algorithm='dense',**kwargs):
    """Compile/load the exact Numba argument-type specialization, without running it.

    Array extents are runtime values for Numba; callers additionally prewarm their
    exact walker/cell/time/state shapes outside performance measurements.
    """
    if u.dtype!=torch.float64 or u.device.type!='cpu':raise ValueError('Requires CPU float64')
    F,G,Q,C,_,_=model_components(u,y,variance,observed,X,model,**kwargs)
    n,d=F.shape[-2:]
    yy=torch.where(observed,y,torch.zeros_like(y));vv=torch.where(observed,variance,torch.ones_like(variance))
    arrays=[yy.numpy().reshape(-1,n),vv.numpy().reshape(-1,n),observed.numpy().reshape(-1,n),F.numpy().reshape(-1,n,d),G.numpy().reshape(-1,d,d),Q.numpy().reshape(-1,d,d),C.numpy().reshape(-1,d,d)]
    if algorithm not in ('dense','structured'):raise ValueError('Unknown experimental algorithm')
    dispatcher=_batch if algorithm=='dense' else _batch_structured
    signature=tuple(typeof(a) for a in arrays);start=time.perf_counter();dispatcher.compile(signature)
    return dict(compilation_or_cache_load_seconds=time.perf_counter()-start,signature=str(signature),parameter_shape=list(u.shape),data_shape=list(y.shape),state_dimension=d,fastmath=False)


def target_numba(u,y,variance,observed,X,model,*,S=10.,sampling=1.,fixed_variance=None,algorithm='dense'):
    """CPU float64 target with the original Torch prior/matrix construction.

    CPU tensors share NumPy storage at the callback boundary. Observations are
    stored once per cell, never copied once per walker. Compile/prewarm this
    exact callback specialization before measuring a sampling comparison.
    """
    if u.dtype!=torch.float64 or u.device.type!='cpu':
        raise ValueError('Numba prototype requires explicit CPU float64')
    if any(z.device.type!='cpu' or z.dtype!=torch.float64 for z in (y,variance,X)):
        raise ValueError('Numba prototype inputs require CPU float64')
    if observed.dtype!=torch.bool or observed.device.type!='cpu' or y.shape!=variance.shape or y.shape!=observed.shape or y.ndim not in (1,2):
        raise ValueError('Matching one-panel or cell/time arrays and Boolean mask required')
    if not bool(torch.isfinite(y[observed]).all() and torch.isfinite(variance[observed]).all() and (variance[observed]>=0).all()):
        raise ValueError('Invalid observed data/variance')
    F,G,Q,C,prior,valid=model_components(u,y,variance,observed,X,model,S=S,sampling=sampling,fixed_variance=fixed_variance)
    shape=F.shape[:-2];n,d=F.shape[-2:]
    def array(z,tail):return z.detach().numpy().reshape((-1,*tail))
    # Missing values are sanitized before entering compiled arithmetic.
    yy=torch.where(observed,y,torch.zeros_like(y));vv=torch.where(observed,variance,torch.ones_like(variance))
    if algorithm not in ('dense','structured'):raise ValueError('Unknown experimental algorithm')
    dispatcher=_batch if algorithm=='dense' else _batch_structured
    ll,failed=dispatcher(array(yy,(n,)),array(vv,(n,)),array(observed,(n,)),array(F,(n,d)),array(G,(d,d)),array(Q,(d,d)),array(C,(d,d)))
    if np.any(failed>=0):raise FloatingPointError(f'Invalid innovation variance at time {int(failed[failed>=0][0])}; no floor')
    likelihood=torch.from_numpy(ll.reshape(shape))
    value=likelihood+prior
    if bool((valid&~torch.isfinite(value)).any()):raise FloatingPointError('Nonfinite density inside parameter support')
    return torch.where(valid,value,torch.full_like(value,-torch.inf))
