"""Shared transformed Torch density for the registered DLM model presets."""
import math
import torch
from .api import _MODEL_SPECS
from .transforms import parameter_names
from .precision_prototype import filter_tensor


def model_components(u,y,variance,observed,X,model,*,S=10.,sampling=1.,fixed_variance=None):
    """Evaluate explicit-mask target with normalized inputs and fixed priors.

    Leading axes broadcast; final parameter axis is ordered by parameter_names.
    No state draws, numerical floors, mixed precision or hidden prior changes.
    """
    spec=_MODEL_SPECS[model];k=X.shape[-1];n=X.shape[-2]
    if any(x.dtype != u.dtype or x.device != u.device for x in (y, variance, X)):
        raise ValueError('Target inputs require a common floating dtype/device')
    if (spec['regressors']=='none' and k!=0) or (spec['regressors']!='none' and k<1):
        raise ValueError('Regressor count does not match model')
    names=parameter_names(model,k)
    if u.shape[-1]!=len(names): raise ValueError('Wrong transformed dimension')
    theta={};lp=torch.zeros_like(u[...,0]);valid=torch.isfinite(u).all(-1)
    for i,name in enumerate(names):
        v=u[...,i]
        if name.startswith('sigma'):
            value=torch.exp(v)
            scale=1e-4 if name.startswith('sigma_reg') or name=='sigma_trend' else (.01 if name=='sigma_seas' else .5)
            lp=lp+v+.5*math.log(2/math.pi)-math.log(scale)-.5*(value/scale)**2
            valid=valid&torch.isfinite(value)&(value>0)
        else:
            width=(.5 if spec['ar2'] else 1.) if name=='rhoAR1' else 1-theta['rhoAR1']
            p=torch.sigmoid(v);value=width*p
            # Normalized uniform physical prior cancels width in Jacobian.
            lp=lp-torch.nn.functional.softplus(-v)-torch.nn.functional.softplus(v)
            valid=valid&(p>0)&(p<1)
        theta[name]=value
    valid=valid&torch.isfinite(lp)
    # Invalid parameter states are evaluated at a harmless interior placeholder
    # then returned as -inf. This is support rejection, not a density correction.
    theta={key:torch.where(valid,value,torch.full_like(value,.1)) for key,value in theta.items()}
    base=6 if spec['seasonal'] else 2;ar=base+k;d=ar+(2 if spec['ar2'] else 1)
    shape=torch.broadcast_shapes(u.shape[:-1],y.shape[:-1],X.shape[:-2])
    G=torch.eye(d,dtype=u.dtype,device=u.device).expand(*shape,d,d).clone()
    G[...,0,1]=1
    Q=u.new_zeros((*shape,d,d));C=G.new_zeros(G.shape)
    for i in range(d):C[...,i,i]=S
    F=u.new_zeros((*shape,n,d));F[...,0]=1;F[...,ar]=1;F[...,base:ar]=X
    Q[...,0,0]=1e-10 if fixed_variance is None else fixed_variance
    Q[...,1,1]=theta['sigma_trend']**2
    if spec['seasonal']:
        for i,period in [(2,12),(4,6)]:
            angle=u.new_tensor(sampling*2*math.pi/period);c=torch.cos(angle);s=torch.sin(angle)
            G[...,i,i]=c;G[...,i,i+1]=s;G[...,i+1,i]=-s;G[...,i+1,i+1]=c
            F[...,i]=1
            Q[...,i,i]=Q[...,i+1,i+1]=theta['sigma_seas']**2
    for j in range(k):
        if spec['regressors']=='dynamic':value=theta[f'sigma_reg[{j}]']**2
        else:value=(1e-20 if spec['ar2'] else 1e-10) if fixed_variance is None else fixed_variance
        Q[...,base+j,base+j]=value
    a=theta['rhoAR1'];sig=theta['sigma_AR'];G[...,ar,ar]=a;Q[...,ar,ar]=sig**2
    if spec['ar2']:
        b=theta['rhoAR2'];G[...,ar,ar+1]=b;G[...,ar+1,ar]=1;G[...,ar+1,ar+1]=0
        Q[...,ar+1,ar+1]=1e-20 if fixed_variance is None else fixed_variance
        v=sig**2/(1-a*a-b*b-2*b*a*a/(1-b))
        C[...,ar,ar]=C[...,ar+1,ar+1]=v
        C[...,ar,ar+1]=C[...,ar+1,ar]=v*a/(1-b)
    else:C[...,ar,ar]=sig**2/(1-a*a)
    return F,G,Q,C,lp,valid


def target(u,y,variance,observed,X,model,*,S=10.,sampling=1.,fixed_variance=None):
    """Deterministic transformed density; shares matrices with reconstruction."""
    F,G,Q,C,lp,valid=model_components(u,y,variance,observed,X,model,S=S,sampling=sampling,fixed_variance=fixed_variance)
    ll=filter_tensor(y,variance,observed,F,G,Q,C)[0].sum(-1)
    value=ll+lp
    if bool((valid&~torch.isfinite(value)).any()):raise FloatingPointError('Nonfinite density inside parameter support')
    return torch.where(valid,value,torch.full_like(value,-torch.inf))
