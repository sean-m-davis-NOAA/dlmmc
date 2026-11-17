"""Vectorized offline physical parameters for six-model diagnostic records."""
import numpy as np
from scipy.special import expit
from .components import component_spec
from .transforms import parameter_names


def physical_parameter_draws(model,u,nreg):
    u=np.asarray(u);names=parameter_names(model,nreg);spec=component_spec(model)
    if u.shape[-1]!=len(names) or not np.isfinite(u).all():raise ValueError('Finite exact-dimension transformed draws required')
    theta={}
    for j,name in enumerate(names):
        if name.startswith('sigma'):value=np.exp(u[...,j])
        else:
            width=(.5 if spec['ar2'] else 1.) if name=='rhoAR1' else 1-theta['rhoAR1']
            value=width*expit(u[...,j])
            if np.any((value<=0)|(value>=width)):raise ValueError('Represented parameter outside strict AR support')
        if not np.isfinite(value).all() or np.any(value<=0):raise ValueError('Represented physical parameter outside positive support')
        theta[name]=value
    return np.stack([theta[name] for name in names],axis=-1)


def stationary_ar_variance(model,physical,nreg):
    names=parameter_names(model,nreg);x=np.asarray(physical)
    rho=x[...,names.index('rhoAR1')];sig=x[...,names.index('sigma_AR')]
    denominator=1-rho*rho
    if component_spec(model)['ar2']:
        b=x[...,names.index('rhoAR2')];denominator=denominator-b*b-2*b*rho*rho/(1-b)
    if not np.isfinite(denominator).all() or np.any(denominator<=0):raise ValueError('Invalid stationary AR denominator')
    return sig*sig/denominator
