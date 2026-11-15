"""Conditional affine sweeps for independent sigma-trend and ENSO-lag fields.

Both field moves are joint over cells, conditional on the other field. Only the
remaining nuisance block factorizes across panels. No graph coloring is used.
"""
import time
import numpy as np
import torch
from affine.torch_stream import State,initialize,_rng_get,_rng_set


def initialize_lag_blocks(target,halves,*,seed,warmup,thin,provenance):
    def callback(flat):
        local=target.local(flat.reshape(-1,target.prepared.cells,target.parameters))
        return target.joint(flat,local),local
    state=initialize(callback,halves,seed=seed,warmup=warmup,thin=thin,save_extras=True,provenance=provenance)
    shape=(2,halves[0].shape[0],len(target.field_indices)+target.prepared.cells)
    for name in ('accepted','proposed','warmup_accepted','warmup_proposed'):
        setattr(state,name,torch.zeros(shape,dtype=torch.int64,device=halves[0].device))
    return state


@torch.no_grad()
def advance_lag_blocks(target,state,transitions,*,deadline_unix=None):
    if transitions<1 or int(transitions)!=transitions or state.extras is None:raise ValueError('Positive transition count and cached local terms required')
    halves=[v.clone() for v in state.halves];lp=[v.clone() for v in state.logp];local=[v.clone() for v in state.extras]
    accepted=state.accepted.clone();proposed=state.proposed.clone();wa=state.warmup_accepted.clone();wp=state.warmup_proposed.clone()
    w,dimension=halves[0].shape;g=target.prepared.cells;p=target.parameters
    fields=tuple(target.field_indices);nuisance=[i for i in range(p) if i not in fields]
    if not nuisance or len(fields)!=2:raise ValueError('Two fields and nonempty conditionally local nuisance required')
    device=halves[0].device;dtype=halves[0].dtype;iteration=state.iteration
    draws=[];logs=[];extras=[];steps=[];old_rng=_rng_get(device)
    try:
        _rng_set(device,state.rng)
        for step in range(state.iteration+1,state.iteration+int(transitions)+1):
            if deadline_unix is not None and time.time()>=deadline_unix:break
            iteration=step;ca,cp=(wa,wp) if step<=state.warmup else (accepted,proposed)
            for field_index,parameter in enumerate(fields):
                for half in (0,1):
                    current=halves[half].reshape(w,g,p);partner=halves[1-half].reshape(w,g,p)[torch.randint(w,(w,),device=device)]
                    z=.5*(torch.rand(w,dtype=dtype,device=device)+1)**2
                    candidate=current.clone();candidate[...,parameter]=partner[...,parameter]+z[:,None]*(current[...,parameter]-partner[...,parameter])
                    candidate_local=target.local(candidate);flat=candidate.reshape(w,dimension);candidate_lp=target.joint(flat,candidate_local)
                    if bool(torch.isnan(candidate_lp).any() or torch.isposinf(candidate_lp).any()):raise FloatingPointError('Invalid field proposal density')
                    decision=torch.log(torch.rand(w,dtype=dtype,device=device))<(g-1)*torch.log(z)+candidate_lp-lp[half]
                    halves[half]=torch.where(decision[:,None],flat,halves[half]);local[half]=torch.where(decision[:,None],candidate_local,local[half]);lp[half]=torch.where(decision,candidate_lp,lp[half])
                    ca[half,:,field_index]+=decision;cp[half,:,field_index]+=1
            for half in (0,1):
                current=halves[half].reshape(w,g,p);partner=halves[1-half].reshape(w,g,p)[torch.randint(w,(w,),device=device)]
                z=.5*(torch.rand((w,g),dtype=dtype,device=device)+1)**2
                candidate=current.clone();candidate[...,nuisance]=partner[...,nuisance]+z[...,None]*(current[...,nuisance]-partner[...,nuisance])
                candidate_local=target.local(candidate)
                if bool(torch.isnan(candidate_local).any() or torch.isposinf(candidate_local).any()):raise FloatingPointError('Invalid local proposal density')
                decision=torch.log(torch.rand((w,g),dtype=dtype,device=device))<(len(nuisance)-1)*torch.log(z)+candidate_local-local[half]
                halves[half]=torch.where(decision[...,None],candidate,current).reshape(w,dimension)
                local[half]=torch.where(decision,candidate_local,local[half]);lp[half]=target.joint(halves[half],local[half])
                ca[half,:,len(fields):]+=decision;cp[half,:,len(fields):]+=1
            if step>state.warmup and (step-state.warmup-1)%state.thin==0:
                draws.append(torch.cat(halves));logs.append(torch.cat(lp));extras.append(torch.cat(local));steps.append(step)
        rng=_rng_get(device).clone()
    finally:_rng_set(device,old_rng)
    result=State(tuple(halves),tuple(lp),tuple(local),iteration,rng,accepted,proposed,wa,wp,state.warmup,state.thin,state.provenance)
    return result,(torch.stack(draws) if draws else halves[0].new_empty((0,2*w,dimension))),(torch.stack(logs) if logs else lp[0].new_empty((0,2*w))),(torch.stack(extras) if extras else local[0].new_empty((0,2*w,g))),np.asarray(steps,dtype=np.int64)
