"""Offline dependence-preserving empirical bootstrap for dtype comparisons.

Whole time blocks include every selected walker and quantity. Independent runs
are strata, not walkers. Weighted order statistics exactly reproduce linear
NumPy quantiles of the resampled empirical draws without duplicating arrays.
No scientific tolerance or numerical-production policy is defined here.
"""
import numpy as np


def block_counts(runs, blocks_per_run, replicates, rng):
    if runs < 4 or blocks_per_run < 5 or replicates < 100:
        raise ValueError('Four independent runs, five blocks/run and >=100 replicates required')
    return np.concatenate([rng.multinomial(blocks_per_run, np.full(blocks_per_run,1/blocks_per_run),size=replicates)
                           for _ in range(runs)],axis=1)


def resampled_quantiles(blocks, counts, probabilities=(.05,.5,.95)):
    """Exact linear quantiles for equal-size blocks repeated by integer counts."""
    blocks=np.asarray(blocks,dtype=np.float64);counts=np.asarray(counts)
    if blocks.ndim!=2 or counts.ndim!=2 or counts.shape[1]!=len(blocks):
        raise ValueError('Expected [block,draw] and [replicate,block]')
    if not np.isfinite(blocks).all() or np.any(counts<0) or not np.all(counts==np.floor(counts)):
        raise ValueError('Finite observations and integer nonnegative counts required')
    if not np.all(counts.sum(1)==len(blocks)):raise ValueError('Bootstrap must retain the full sample count')
    q=np.asarray(probabilities)
    if np.any((q<0)|(q>1)):raise ValueError('Invalid quantile')
    ordered=np.sort(blocks,axis=1);pooled=np.sort(blocks.ravel())
    position=q*(len(pooled)-1);lowrank=np.floor(position).astype(np.int64);highrank=np.ceil(position).astype(np.int64)
    ranks=np.concatenate([lowrank,highrank]);width=len(ranks);b=len(counts)
    left=np.zeros((b,width),dtype=np.int64);right=np.full_like(left,len(pooled)-1)
    # Find the smallest original value with weighted CDF count > target rank.
    while np.any(left<right):
        middle=(left+right)//2;value=pooled[middle]
        cdf=np.zeros_like(middle)
        for j,values in enumerate(ordered):cdf+=np.searchsorted(values,value,side='right')*counts[:,j,None]
        above=cdf>ranks[None,:]
        right=np.where(above,middle,right);left=np.where(above,left,middle+1)
    values=pooled[left];k=len(q);fraction=position-lowrank
    return values[:,:k]*(1-fraction)+values[:,k:]*fraction


def bootstrap_metrics(draws, counts, block_steps, *, quantiles=(.05,.5,.95), progress=None):
    """Metrics in [mean, SD, q05, q50, q95] order for every quantity.

    draws: [independent run,time,selected walker,quantity]. No suffix truncation.
    """
    x=np.asarray(draws,dtype=np.float64)
    if x.ndim!=4 or x.shape[0]<4 or not np.isfinite(x).all():raise ValueError('Four finite independent runs required')
    r,t,w,p=x.shape
    if t%block_steps or t//block_steps<5:raise ValueError('Complete blocks required; no discarded suffix')
    n=r*t*w;blocks=x.reshape(r*(t//block_steps),block_steps*w,p)
    if counts.shape[1]!=len(blocks):raise ValueError('Block count mismatch')
    if not np.all(counts.reshape(len(counts),r,-1).sum(2)==t//block_steps):raise ValueError('Each run must keep its sample weight')
    center=x.mean((0,1,2));delta=blocks-center
    sums=delta.sum(1);squares=(delta*delta).sum(1)
    means=counts@sums/n;second=counts@squares
    variance=(second-n*means*means)/(n-1)
    if np.any(variance<0):raise FloatingPointError('Negative resampled variance; no clipping')
    boot=np.empty((len(counts),2+len(quantiles),p));boot[:,0]=means+center;boot[:,1]=np.sqrt(variance)
    point=np.vstack([center,x.std((0,1,2),ddof=1),np.quantile(x,quantiles,axis=(0,1,2))])
    for j in range(p):
        boot[:,2:,j]=resampled_quantiles(blocks[:,:,j],counts,quantiles)
        if progress is not None:progress(j,p)
    return point,boot


def bootstrap_probabilities(indicators,counts,block_steps):
    x=np.asarray(indicators,dtype=float)
    if x.ndim!=4 or np.any((x<0)|(x>1)) or not np.isfinite(x).all():raise ValueError('Finite probabilities/indicators required')
    r,t,w,p=x.shape
    if t%block_steps or counts.shape[1]!=r*t//block_steps:raise ValueError('Complete matching blocks required')
    b=x.reshape(r*t//block_steps,block_steps*w,p).mean(1)
    return x.mean((0,1,2)),counts@b/len(b)


def discrepancy_intervals(point32,point64,bootstrap32,bootstrap64,confidence=.95):
    """Basic marginal and max-standardized simultaneous bootstrap MC intervals."""
    delta=np.asarray(point32)-np.asarray(point64)
    differences=np.asarray(bootstrap32)-np.asarray(bootstrap64)
    if differences.ndim!=2 or delta.shape!=differences.shape[1:]:raise ValueError('Flattened metrics required')
    error=differences-delta
    se=differences.std(0,ddof=1)
    if np.any(se<=0) or not np.isfinite(se).all():raise ValueError('Degenerate bootstrap uncertainty; no SE floor')
    alpha=1-confidence
    q=np.quantile(error,[alpha/2,1-alpha/2],axis=0)
    pointwise=np.stack([delta-q[1],delta-q[0]],axis=-1)
    maximum=np.max(np.abs(error/se),axis=1)
    critical=float(np.quantile(maximum,confidence))
    simultaneous=np.stack([delta-critical*se,delta+critical*se],axis=-1)
    return dict(difference=delta.tolist(),bootstrap_se=se.tolist(),absolute_difference_over_se=(np.abs(delta)/se).tolist(),
                within_one_se=(np.abs(delta)<=se).tolist(),pointwise_interval=pointwise.tolist(),simultaneous_interval=simultaneous.tolist(),
                pointwise_detected=((pointwise[:,0]>0)|(pointwise[:,1]<0)).tolist(),
                simultaneous_detected=((simultaneous[:,0]>0)|(simultaneous[:,1]<0)).tolist(),
                simultaneous_critical=critical,confidence=confidence,replicates=len(differences))
