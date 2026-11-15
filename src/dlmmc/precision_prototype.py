"""Native tensor Joseph-filter prototype, not a qualified production API."""
import torch


def filter_tensor(y, variance, observed, F, G, Q, C):
    """Broadcast leading axes, time penultimate in F; final axes are states.

    Inputs are prepared in the requested dtype/device. Invalid observed data
    fail explicitly; masked values are sanitized before any numerical work.
    """
    if any(x.dtype!=G.dtype or x.device!=G.device for x in (y,variance,F,Q,C)):
        raise ValueError('Mixed dtype/device is not supported')
    if observed.dtype!=torch.bool or observed.device!=G.device:
        raise ValueError('Boolean mask must reside on the input device')
    if not bool(torch.isfinite(y[observed]).all() and torch.isfinite(variance[observed]).all() and (variance[observed]>=0).all()):
        raise ValueError('Invalid observed data/variance')
    y=torch.where(observed,y,torch.zeros_like(y))
    variance=torch.where(observed,variance,torch.ones_like(variance))
    n,d=F.shape[-2:]
    shape=torch.broadcast_shapes(G.shape[:-2],F.shape[:-2],y.shape[:-1])
    P=C.expand(*shape,d,d).clone();m=G.new_zeros((*shape,d))
    eye=torch.eye(d,dtype=G.dtype,device=G.device)
    terms=[]; means=[]; covariances=[]
    for t in range(n):
        m=(G@m.unsqueeze(-1)).squeeze(-1)
        P=G@P@G.transpose(-1,-2)+Q
        f=F[...,t,:]; p=(P@f.unsqueeze(-1)).squeeze(-1)
        s=(f*p).sum(-1)+variance[...,t]
        mask=observed[...,t]
        if not bool(torch.isfinite(s).all() and (s>0).all()):
            raise ValueError(f'Invalid innovation variance at time {t}')
        v=y[...,t]-(f*m).sum(-1)
        k=p/s.unsqueeze(-1)
        A=eye-k.unsqueeze(-1)*f.unsqueeze(-2)
        updated=A@P@A.transpose(-1,-2)+(k.unsqueeze(-1)*k.unsqueeze(-2))*variance[...,t,None,None]
        m=torch.where(mask[...,None],m+k*v.unsqueeze(-1),m)
        P=torch.where(mask[...,None,None],updated,P)
        terms.append(torch.where(mask,-.5*(torch.log(2*torch.pi*s)+v*v/s),torch.zeros_like(s)))
        means.append(m);covariances.append(P)
    return torch.stack(terms,-1),torch.stack(means,-2),torch.stack(covariances,-3)
