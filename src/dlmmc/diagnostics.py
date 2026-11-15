"""Dependence-aware pilot diagnostics; walkers are never separate chains."""
import numpy as np


def autocorrelation_time(series):
    """Initial-positive, initial-monotone paired autocorrelation estimate."""
    x=np.asarray(series,dtype=float)
    if x.ndim!=1 or len(x)<8 or not np.isfinite(x).all():
        raise ValueError('At least eight finite ordered observations required')
    x=x-x.mean();n=len(x)
    if np.dot(x,x)==0:return np.inf
    ft=np.fft.rfft(x,n=2**int(np.ceil(np.log2(2*n))))
    ac=np.fft.irfft(ft*ft.conj())[:n]
    ac=ac/ac[0]
    pairs=ac[:2*(n//2)].reshape(-1,2).sum(1)
    stop=np.flatnonzero(pairs<=0)
    if len(stop):pairs=pairs[:stop[0]]
    pairs=np.minimum.accumulate(pairs)
    return max(float(-1+2*pairs.sum()),np.finfo(float).eps)


def split_rhat(x):
    """Unranked split Rhat for actual independent run time series."""
    x=np.asarray(x,dtype=float);n=x.shape[1]//2
    z=np.concatenate((x[:,:n],x[:,-n:]),axis=0)
    within=z.var(axis=1,ddof=1).mean(axis=0)
    between=n*z.mean(axis=1).var(axis=0,ddof=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.sqrt(((n-1)/n*within+between/n)/within)


def ensemble_diagnostics(draws):
    """Input [independent run, step, walker, quantity].

    Estimate mean precision through the autocovariance of per-step ensemble
    averages, including cross-walker lag dependence. Short estimates are
    explicitly unreliable. This is not a quantile/tail or convergence gate.
    """
    x=np.asarray(draws,dtype=float)
    if x.ndim!=4 or x.shape[0]<4 or x.shape[1]<8 or not np.isfinite(x).all():
        raise ValueError('Four finite independent ensembles with ordered steps required')
    r,t,w,p=x.shape;average=x.mean(2)
    tau=np.array([[autocorrelation_time(average[i,:,j]) for j in range(p)] for i in range(r)])
    run_var=np.full_like(tau,np.inf)
    finite=np.isfinite(tau)
    run_var[finite]=average.var(1,ddof=1)[finite]*tau[finite]/t
    mcvar=run_var.sum(0)/r**2
    posterior_var=x.reshape(-1,p).var(0,ddof=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        ess=np.minimum(r*t*w,posterior_var/mcvar)
    return dict(iat_by_run=tau.tolist(),mean_ess_estimate=ess.tolist(),mean_mcse=np.sqrt(mcvar).tolist(),iat_length_reliable=(t>=50*tau.max(0)).tolist(),split_rhat_ensemble_means=split_rhat(average).tolist(),retained_steps=t,independent_runs=r,walkers=w,status='ESTIMATE_NOT_CONVERGENCE_CERTIFICATE')


def rank_split_rhat(x):
    """Rank-normalized and folded split Rhat across independent runs."""
    from scipy.stats import rankdata
    from scipy.special import ndtri
    x=np.asarray(x,dtype=float)
    n=x.shape[0]*x.shape[1]
    rank=rankdata(x.reshape(n,-1),axis=0)
    z=ndtri((rank-.375)/(n+.25)).reshape(x.shape)
    folded=np.abs(x-np.median(x,axis=(0,1)))
    rank_fold=rankdata(folded.reshape(n,-1),axis=0)
    zfold=ndtri((rank_fold-.375)/(n+.25)).reshape(x.shape)
    return np.maximum(split_rhat(z),split_rhat(zfold))
