"""Reference transforms with explicit Jacobians for constrained DLM priors."""
import numpy as np
from scipy.special import expit
from .components import compiled_layout, component_spec


def parameter_names(model, nreg):
    return list(compiled_layout(model,0 if component_spec(model)['regressors']=='none' else nreg).parameters)


def unconstrain(model,theta,nreg):
    """Invert the reference transform without clipping support boundaries."""
    spec=component_spec(model);values=[]
    for name in parameter_names(model,nreg):
        if name.startswith('sigma_reg'):
            value=theta['sigma_reg'][int(name.split('[')[1][:-1])]
        else:value=theta[name]
        if not np.isfinite(value):raise ValueError('Nonfinite physical parameter')
        if name.startswith('sigma'):
            if value<=0:raise ValueError('Scale outside positive support')
            values.append(np.log(value))
        else:
            width=(.5 if spec['ar2'] else 1.) if name=='rhoAR1' else 1-theta['rhoAR1']
            if not 0<value<width:raise ValueError('AR parameter outside conditional support')
            p=value/width;values.append(np.log(p)-np.log1p(-p))
    return np.asarray(values)


def constrain(model, u, nreg):
    """Log scales, logit rho1, and conditional logit rho2/(1-rho1)."""
    spec=component_spec(model);names=parameter_names(model,nreg)
    u=np.asarray(u)
    if u.ndim!=1 or len(u)!=len(names) or not np.isfinite(u).all():
        raise ValueError('Finite parameter vector with exact model dimension required')
    out={};logj=0.;regs=[]
    for i,name in enumerate(names):
        v=u[i]
        if name.startswith('sigma'):
            value=np.exp(v);logj+=v
            if name.startswith('sigma_reg'):
                out[name]=value;regs.append((int(name.split('[')[1][:-1]),value))
            else:out[name]=value
        else:
            width=(.5 if spec['ar2'] else 1.) if name=='rhoAR1' else 1-out['rhoAR1']
            out[name]=width*expit(v)
            logj+=np.log(width)-np.logaddexp(0.,-v)-np.logaddexp(0.,v)
    if regs:
        out['sigma_reg']=np.zeros(nreg)
        for j,value in regs:out['sigma_reg'][j]=value
    return out,logj
