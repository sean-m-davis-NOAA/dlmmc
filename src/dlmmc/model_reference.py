"""Auditable NumPy matrices for the six pinned Stan models (offline reference)."""
import numpy as np
from .api import _MODEL_SPECS


def matrices(model, regressors, theta, *, S=10., sampling=1., dtype=np.float64, fixed_variance=None):
    spec = _MODEL_SPECS[model]
    X = np.asarray(regressors, dtype=dtype)
    n, k = X.shape
    if spec['regressors'] == 'none' and k:
        raise ValueError('noregs model requires zero regressor columns')
    base = 6 if spec['seasonal'] else 2
    ar = base+k
    d = ar + (2 if spec['ar2'] else 1)
    G = np.eye(d, dtype=dtype); G[0,1] = 1
    Q = np.zeros((d,d), dtype=dtype); C = np.eye(d,dtype=dtype)*dtype(S)
    F = np.zeros((n,d),dtype=dtype); F[:,0] = 1; F[:,ar] = 1
    Q[0,0] = 1e-10 if fixed_variance is None else fixed_variance
    Q[1,1] = dtype(theta['sigma_trend'])**2
    if spec['seasonal']:
        for i, period in [(2,12),(4,6)]:
            angle=dtype(sampling)*dtype(2*np.pi/period)
            c,s=np.cos(angle),np.sin(angle)
            G[i:i+2,i:i+2]=[[c,s],[-s,c]]
            F[:,i]=1
            Q[i,i]=Q[i+1,i+1]=dtype(theta['sigma_seas'])**2
    F[:,base:ar]=X
    for j in range(k):
        if spec['regressors']=='dynamic':
            Q[base+j,base+j]=dtype(theta['sigma_reg'][j])**2
        else:
            Q[base+j,base+j]=(1e-20 if spec['ar2'] else 1e-10) if fixed_variance is None else fixed_variance
    a=dtype(theta['rhoAR1']); sig=dtype(theta['sigma_AR'])
    if not 0<a<(0.5 if spec['ar2'] else 1):
        raise ValueError('AR1 outside support')
    G[ar,ar]=a; Q[ar,ar]=sig**2
    if spec['ar2']:
        b=dtype(theta['rhoAR2'])
        if not 0<b<1-a: raise ValueError('AR2 outside conditional support')
        G[ar,ar+1]=b; G[ar+1,ar]=1; G[ar+1,ar+1]=0
        Q[ar+1,ar+1]=1e-20 if fixed_variance is None else fixed_variance
        v=sig**2/(1-a*a-b*b-2*b*a*a/(1-b))
        C[ar:,ar:]=[[v,v*a/(1-b)],[v*a/(1-b),v]]
    else:
        C[ar,ar]=sig**2/(1-a*a)
    return F,G,Q,C


def physical_logprior(model, theta, scales):
    """Normalized physical priors, including conditional AR2 uniform density."""
    spec=_MODEL_SPECS[model]
    keys=['sigma_trend','sigma_AR']+(['sigma_seas'] if spec['seasonal'] else [])
    if spec['regressors']=='dynamic': keys+=['sigma_reg']
    lp=0.
    for key in keys:
        x=np.asarray(theta[key]); s=np.asarray(scales[key+'_prior'])
        if np.any(x<=0) or not np.all(np.isfinite(x)): return -np.inf
        lp+=np.sum(.5*np.log(2/np.pi)-np.log(s)-.5*(x/s)**2)
    a=theta['rhoAR1']
    if not 0<a<(0.5 if spec['ar2'] else 1): return -np.inf
    if spec['ar2']:
        if not 0<theta['rhoAR2']<1-a: return -np.inf
        lp+=np.log(2)-np.log1p(-a)
    return lp
