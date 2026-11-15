"""Prespecified physical-unit summaries for scientific comparisons.

Call only on retained, diagnosed parameter/state draws (or fixed test fixtures).
State index zero is x1, after the first transition, matching the pinned model.
"""
import numpy as np
from .api import _MODEL_SPECS
from .transforms import parameter_names


def scientific_summaries(model,parameters,states,*,nreg,scale,offset,time_indices,contrasts,amplitude_times):
    """Return named scalar summaries and named positive-contrast indicators.

    Parameters are ordered physical model parameters in normalized model units;
    states have trailing [time,state] axes. All preceding draw axes must match.
    The fixed scaling/centering constants come from data preparation provenance.
    """
    spec=_MODEL_SPECS[model];p=np.asarray(parameters);x=np.asarray(states)
    names=parameter_names(model,nreg)
    if p.shape[-1]!=len(names) or p.shape[:-1]!=x.shape[:-2]:
        raise ValueError('Parameter/state draw axes do not match')
    if not np.isfinite(p).all() or not np.isfinite(x).all():raise ValueError('Nonfinite draws cannot be discarded')
    if not np.isfinite(scale) or scale<=0 or not np.isfinite(offset):raise ValueError('Invalid fixed normalization constants')
    base=6 if spec['seasonal'] else 2;ar=base+nreg;expected=ar+(2 if spec['ar2'] else 1)
    if x.shape[-1]!=expected:raise ValueError('State dimension does not match model')
    n=x.shape[-2]
    if any(t<0 or t>=n for t in list(time_indices)+list(amplitude_times)) or any(not 0<=a<b<n for a,b in contrasts):
        raise ValueError('Prespecified time index outside the trajectory')
    result={}
    for i,name in enumerate(names):result[name]=p[...,i]*scale if name.startswith('sigma') else p[...,i]
    theta={name:p[...,i] for i,name in enumerate(names)}
    rho=theta['rhoAR1'];sig=theta['sigma_AR']
    denominator=1-rho**2
    if spec['ar2']:
        b=theta['rhoAR2'];denominator=denominator-b**2-2*b*rho**2/(1-b)
    if np.any(denominator<=0):raise ValueError('Nonstationary AR draw')
    result['stationary_AR_variance']=sig**2/denominator*scale**2
    for t in time_indices:
        result[f'trend[{t}]']=x[...,t,0]*scale+offset
        result[f'slope[{t}]']=x[...,t,1]*scale
        if spec['seasonal']:result[f'seasonal_sum[{t}]']=(x[...,t,2]+x[...,t,4])*scale
        result[f'AR_component[{t}]']=x[...,t,ar]*scale
    for j in range(nreg):
        for t in (time_indices if spec['regressors']=='dynamic' else [0]):
            result[f'beta[{j},{t}]']=x[...,t,base+j]*scale
    probabilities={}
    for a,b in contrasts:
        contrast=(x[...,b,0]-x[...,a,0])*scale
        result[f'trend_change[{a},{b}]']=contrast
        probabilities[f'trend_positive[{a},{b}]']=(contrast>0)
    result['average_slope']=(x[...,-1,0]-x[...,0,0])*scale/(n-1)
    if spec['seasonal']:
        for t in amplitude_times:
            result[f'annual_amplitude[{t}]']=np.hypot(x[...,t,2],x[...,t,3])*scale
            result[f'semiannual_amplitude[{t}]']=np.hypot(x[...,t,4],x[...,t,5])*scale
    return list(result),np.stack(list(result.values()),axis=-1),list(probabilities),np.stack(list(probabilities.values()),axis=-1)
