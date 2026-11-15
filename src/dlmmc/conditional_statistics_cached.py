"""Reuse identical RTS moments and Joseph innovations across trend contrasts.

This computes the same scalar operations, in the same per-contrast order, as
contrast_moments. Caching removes repeated solves; it does not thin draws or
change the Gaussian conditional, variance arithmetic, CDF or scientific target.
"""
import torch
from .tensor_smoothing import conditional_moments
from .conditional_statistics_stable import normal_positive_probability


def cached_positive_trend_probabilities(filtered,G,Q,contrasts):
    m,c,j=conditional_moments(filtered,G,Q)
    left=G.new_zeros(G.shape[-1]);left[0]=-1;right=-left
    eye=torch.eye(G.shape[-1],dtype=G.dtype,device=G.device)
    innovation_cache={};result=[]
    for a,b in contrasts:
        if not 0<=a<b<filtered[1].shape[-2]:raise ValueError('Require ordered, distinct time indices')
        mean=(left*m[...,a,:]).sum(-1)+(right*m[...,b,:]).sum(-1)
        weight=left.expand_as(m[...,a,:]);variance=torch.zeros_like(mean)
        for t in range(a,b):
            gain=j[...,t,:,:]
            if t not in innovation_cache:
                A=eye-gain@G
                innovation_cache[t]=A@filtered[2][...,t,:,:]@A.transpose(-1,-2)+gain@Q@gain.transpose(-1,-2)
            conditional=innovation_cache[t]
            variance=variance+(weight*(conditional@weight.unsqueeze(-1)).squeeze(-1)).sum(-1)
            weight=(gain.transpose(-1,-2)@weight.unsqueeze(-1)).squeeze(-1)
        weight=weight+right
        variance=variance+(weight*(c[...,b,:,:]@weight.unsqueeze(-1)).squeeze(-1)).sum(-1)
        if not bool(torch.isfinite(variance).all() and (variance>=0).all()):raise FloatingPointError('Invalid conditional contrast variance; no clipping')
        result.append(normal_positive_probability(mean,variance))
    return torch.stack(result,-1)
