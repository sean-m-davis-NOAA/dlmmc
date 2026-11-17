"""Current-state-only Joseph filter; no host transfers inside the time loop."""
import torch
from .tensor_target import model_components


def log_likelihood(y,variance,observed,F,G,Q,C):
    y=torch.where(observed,y,torch.zeros_like(y))
    variance=torch.where(observed,variance,torch.ones_like(variance))
    n,d=F.shape[-2:];shape=torch.broadcast_shapes(G.shape[:-2],F.shape[:-2],y.shape[:-1])
    P=C.expand(*shape,d,d).clone();m=G.new_zeros((*shape,d));total=G.new_zeros(shape)
    eye=torch.eye(d,dtype=G.dtype,device=G.device)
    invalid=torch.zeros(shape,dtype=torch.bool,device=G.device)
    for t in range(n):
        m=(G@m.unsqueeze(-1)).squeeze(-1);P=G@P@G.transpose(-1,-2)+Q
        f=F[...,t,:];p=(P@f.unsqueeze(-1)).squeeze(-1)
        s=(f*p).sum(-1)+variance[...,t];mask=observed[...,t]
        invalid=invalid|(~torch.isfinite(s))|(s<=0)
        v=y[...,t]-(f*m).sum(-1);k=p/s.unsqueeze(-1)
        A=eye-k.unsqueeze(-1)*f.unsqueeze(-2)
        updated=A@P@A.transpose(-1,-2)+k.unsqueeze(-1)*k.unsqueeze(-2)*variance[...,t,None,None]
        m=torch.where(mask[...,None],m+k*v.unsqueeze(-1),m)
        P=torch.where(mask[...,None,None],updated,P)
        total=total+torch.where(mask,-.5*(torch.log(2*torch.pi*s)+v*v/s),torch.zeros_like(s))
    if bool(invalid.any() or (~torch.isfinite(total)).any()):
        raise FloatingPointError('Invalid innovation/density; no clipping or numerical floor')
    return total


class TorchTarget:
    """Reuse the validated model specification, prior and Jacobian equations.

    Observation vectors are cached once per cell. A zero-time matrix call builds
    only parameter-dependent G/Q/C, avoiding walker copies of the time axis.
    """
    def __init__(self,prepared):
        self.prepared=prepared;p=prepared
        center=p.y.new_zeros((p.cells,len(p.parameter_names)))
        self.F=model_components(center,*p.arguments,p.model,S=p.priors['S'],sampling=p.sampling)[0]
        self.empty=(p.y[:,:0],p.variance[:,:0],p.observed[:,:0],p.regressors[:,:0])

    def prior_correction(self,u,valid):
        p=self.prepared;value=torch.zeros_like(u[...,0])
        from .components import defaults
        baseline=defaults(p.model,p.regressors.shape[-1])
        for j,name in enumerate(p.parameter_names):
            if not name.startswith('sigma'):continue
            key='sigma_reg_prior' if name.startswith('sigma_reg') else name+'_prior'
            desired=p.priors[key]
            if name.startswith('sigma_reg'):desired=desired[int(name.split('[')[1][:-1])]
            old=baseline[name]
            if float(desired)==old:continue
            theta=torch.exp(torch.where(valid,u[...,j],torch.zeros_like(u[...,j])))
            value=value+torch.log(u.new_tensor(old/float(desired)))+.5*(theta/old)**2-.5*(theta/float(desired))**2
        return value

    def __call__(self,u):
        p=self.prepared
        if u.ndim!=3 or u.shape[1]!=p.cells or u.shape[-1]!=len(p.parameter_names):raise ValueError('Expected [walkers,cells,parameters]')
        _,G,Q,C,prior,valid=model_components(u,*self.empty,p.model,S=p.priors['S'],sampling=p.sampling)
        result=log_likelihood(p.y,p.variance,p.observed,self.F,G,Q,C)+prior+self.prior_correction(u,valid)
        if bool((valid&~torch.isfinite(result)).any()):raise FloatingPointError('Nonfinite valid target')
        return torch.where(valid,result,torch.full_like(result,-torch.inf))
