"""Exact conditional-block affine updates for one correlated sigma field.

Field moves use one acceptance per walker, with moved dimension G. Conditional
on that field, nuisance likelihood/prior factors are independent across cells;
their stretch dimension is P-1. Complementary halves are updated in sequence.
This is dense-Gaussian conditioning, not graph coloring or independent field
acceptance. Cached local densities change only when their proposals are accepted.
"""
import math
import time
import numpy as np
import torch
from affine.torch_stream import State,initialize,_rng_get,_rng_set


def _local(target,flat):
    p=target.prepared
    return target.local(flat.reshape(-1,p.cells,target.parameters))


def initialize_blocks(target,halves,*,seed,warmup,thin,provenance):
    def callback(x):return target(x),_local(target,x)
    state=initialize(callback,halves,seed=seed,warmup=warmup,thin=thin,save_extras=True,provenance=provenance)
    shape=(2,halves[0].shape[0],target.prepared.cells+1)
    counts=torch.zeros(shape,dtype=torch.int64,device=halves[0].device)
    state.accepted=counts.clone();state.proposed=counts.clone();state.warmup_accepted=counts.clone();state.warmup_proposed=counts.clone()
    return state


def _joint(target,flat,local):
    u=flat.reshape(-1,target.prepared.cells,target.parameters);valid=torch.isfinite(local).all(-1)
    sigma=torch.where(valid[:,None],u[...,target.index],torch.zeros_like(u[...,target.index])).exp()
    old=.5*math.log(2/math.pi)-math.log(target.old_scale)-.5*(sigma/target.old_scale)**2
    result=local.sum(-1)-old.sum(-1)+target.field.log_kernel(sigma)
    return torch.where(valid,result,torch.full_like(result,-torch.inf))


@torch.no_grad()
def advance_blocks(target,state,transitions,*,deadline_unix=None):
    if transitions<1 or int(transitions)!=transitions or state.extras is None:raise ValueError('Positive integer transitions and cached local densities required')
    halves=[v.clone() for v in state.halves];lp=[v.clone() for v in state.logp];local=[v.clone() for v in state.extras]
    accepted=state.accepted.clone();proposed=state.proposed.clone();wa=state.warmup_accepted.clone();wp=state.warmup_proposed.clone()
    w,dimension=halves[0].shape;g=target.prepared.cells;p=target.parameters;j=target.index
    nuisance=[k for k in range(p) if k!=j];device=halves[0].device;dtype=halves[0].dtype
    if len(nuisance)<1:raise ValueError('Nuisance block must be nonempty')
    draws=[];logs=[];extras=[];steps=[];old_rng=_rng_get(device)
    iteration=state.iteration
    try:
        _rng_set(device,state.rng)
        for next_iteration in range(state.iteration+1,state.iteration+int(transitions)+1):
            # Check every transition, including after process suspension. A
            # delayed process cannot prevent elapsed-wall overrun, but must not
            # continue an entire 50-transition chunk after its deadline.
            if deadline_unix is not None and time.time()>=deadline_unix:break
            iteration=next_iteration
            counts_a,counts_p=(wa,wp) if iteration<=state.warmup else (accepted,proposed)
            # Joint sigma field: dense coupling requires one joint decision.
            for half in (0,1):
                current=halves[half].reshape(w,g,p);partner=halves[1-half].reshape(w,g,p)[torch.randint(w,(w,),device=device)]
                z=.5*(torch.rand(w,dtype=dtype,device=device)+1)**2
                candidate=current.clone();candidate[...,j]=partner[...,j]+z[:,None]*(current[...,j]-partner[...,j]);flat=candidate.reshape(w,dimension)
                candidate_local=_local(target,flat);candidate_lp=_joint(target,flat,candidate_local)
                if bool(torch.isnan(candidate_lp).any() or torch.isposinf(candidate_lp).any()):raise FloatingPointError('Invalid field proposal density')
                decision=torch.log(torch.rand(w,dtype=dtype,device=device))<(g-1)*torch.log(z)+candidate_lp-lp[half]
                halves[half]=torch.where(decision[:,None],flat,halves[half]);local[half]=torch.where(decision[:,None],candidate_local,local[half]);lp[half]=torch.where(decision,candidate_lp,lp[half])
                counts_a[half,:,0]+=decision;counts_p[half,:,0]+=1
            # Nuisance blocks condition on each walker's unchanged full field.
            for half in (0,1):
                current=halves[half].reshape(w,g,p);partner=halves[1-half].reshape(w,g,p)[torch.randint(w,(w,),device=device)]
                z=.5*(torch.rand((w,g),dtype=dtype,device=device)+1)**2
                candidate=current.clone();candidate[...,nuisance]=partner[...,nuisance]+z[...,None]*(current[...,nuisance]-partner[...,nuisance])
                candidate_local=_local(target,candidate.reshape(w,dimension))
                if bool(torch.isnan(candidate_local).any() or torch.isposinf(candidate_local).any()):raise FloatingPointError('Invalid local proposal density')
                decision=torch.log(torch.rand((w,g),dtype=dtype,device=device))<(len(nuisance)-1)*torch.log(z)+candidate_local-local[half]
                halves[half]=torch.where(decision[...,None],candidate,current).reshape(w,dimension)
                local[half]=torch.where(decision,candidate_local,local[half]);lp[half]=_joint(target,halves[half],local[half])
                counts_a[half,:,1:]+=decision;counts_p[half,:,1:]+=1
            if iteration>state.warmup and (iteration-state.warmup-1)%state.thin==0:
                draws.append(torch.cat(halves));logs.append(torch.cat(lp));extras.append(torch.cat(local));steps.append(iteration)
        rng=_rng_get(device).clone()
    finally:_rng_set(device,old_rng)
    result=State(tuple(halves),tuple(lp),tuple(local),iteration,rng,accepted,proposed,wa,wp,state.warmup,state.thin,state.provenance)
    return result,(torch.stack(draws) if draws else halves[0].new_empty((0,2*w,dimension))), (torch.stack(logs) if logs else lp[0].new_empty((0,2*w))), (torch.stack(extras) if extras else local[0].new_empty((0,2*w,g))),np.asarray(steps,dtype=np.int64)
