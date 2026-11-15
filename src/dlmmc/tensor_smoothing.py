"""Conditional moment prototype; invoke only after posterior diagnostics pass.

Fixed-parameter reference tests are separate from posterior reconstruction.
"""
import torch


def conditional_draw(y,variance,observed,F,G,Q,C,*,generator,filter_function=None):
    """One conditional trajectory per leading draw/cell index.

    Simulation smoothing uses a separate explicit RNG generator. A caller
    reconstructing posterior draws must first qualify its retained window.
    Diagonal PSD process noise and positive-definite initial covariance are
    required; exact zero process/measurement variance needs no artificial floor.
    """
    if filter_function is None:
        from .precision_prototype import filter_tensor
        filter_function=filter_tensor
    if generator is None:
        raise ValueError('Supply an independent reconstruction RNG generator')
    if any(x.dtype!=G.dtype or x.device!=G.device for x in (y,variance,F,Q,C)):
        raise ValueError('Mixed dtype/device is not supported')
    if observed.dtype!=torch.bool or observed.device!=G.device:
        raise ValueError('Boolean mask must reside on the input device')
    if not bool(torch.isfinite(y[observed]).all() and torch.isfinite(variance[observed]).all() and (variance[observed]>=0).all()):
        raise ValueError('Invalid observed data/variance')
    diagonal=Q.diagonal(dim1=-2,dim2=-1)
    if not bool(torch.isfinite(Q).all() and (diagonal>=0).all() and (Q==torch.diag_embed(diagonal)).all()):
        raise ValueError('Simulation smoother requires diagonal PSD process noise')
    y=torch.where(observed,y,torch.zeros_like(y))
    variance=torch.where(observed,variance,torch.zeros_like(variance))
    n,d=F.shape[-2:];shape=torch.broadcast_shapes(G.shape[:-2],F.shape[:-2],y.shape[:-1])
    def normal(tail):
        return torch.randn((*shape,*tail),dtype=G.dtype,device=G.device,generator=generator)
    state=(torch.linalg.cholesky(C)@normal((d,)).unsqueeze(-1)).squeeze(-1)
    process_root=torch.sqrt(diagonal);states=[];pseudo=[]
    for t in range(n):
        state=(G@state.unsqueeze(-1)).squeeze(-1)+process_root*normal((d,))
        states.append(state)
        pseudo.append((F[...,t,:]*state).sum(-1)+torch.sqrt(variance[...,t])*normal(()))
    simulated=torch.stack(states,-2)
    residual=y-torch.stack(pseudo,-1)
    # The residual model has the same covariance and zero initial mean.
    filtered=filter_function(residual,variance.expand_as(residual),observed.expand_as(residual),F,G,Q,C)
    correction=conditional_moments(filtered,G,Q)[0]
    return simulated+correction


def conditional_moments(filtered, G, Q):
    """RTS mean, marginal covariance and backward gains, in native dtype.

    No jitter, clipping or float64 fallback. Gain J_t gives cross-time covariance
    Cov(x_t,x_s|y)=J_t Cov(x_{t+1},x_s|y), for s>t.
    """
    _, m, P=filtered
    means=[m[..., -1, :]];covs=[P[..., -1, :, :]];gains=[]
    for t in range(m.shape[-2]-2,-1,-1):
        prediction=(G@m[...,t,:,None]).squeeze(-1)
        R=G@P[...,t,:,:]@G.transpose(-1,-2)+Q
        J=torch.linalg.solve(R,G@P[...,t,:,:]).transpose(-1,-2)
        means.append(m[...,t,:]+(J@(means[-1]-prediction).unsqueeze(-1)).squeeze(-1))
        # Conditional covariance in Joseph form: no cancellation P - J R J'.
        A=torch.eye(G.shape[-1],dtype=G.dtype,device=G.device)-J@G
        conditional=A@P[...,t,:,:]@A.transpose(-1,-2)+J@Q@J.transpose(-1,-2)
        covs.append(conditional+J@covs[-1]@J.transpose(-1,-2))
        gains.append(J)
    return torch.stack(means[::-1],-2),torch.stack(covs[::-1],-3),torch.stack(gains[::-1],-3) if gains else G.new_empty((*G.shape[:-2],0,*G.shape[-2:]))


def full_conditional_covariance(covariances,gains):
    """Tiny-fixture cross-time covariance; intentionally not a grid API."""
    n,d=covariances.shape[-3],covariances.shape[-1]
    if n>32:raise ValueError('Dense reference covariance restricted to 32 time points')
    out=covariances.new_zeros((*covariances.shape[:-3],n*d,n*d))
    for s in range(n):
        cross=covariances[...,s,:,:]
        out[...,s*d:(s+1)*d,s*d:(s+1)*d]=cross
        for t in range(s-1,-1,-1):
            cross=gains[...,t,:,:]@cross
            out[...,t*d:(t+1)*d,s*d:(s+1)*d]=cross
            out[...,s*d:(s+1)*d,t*d:(t+1)*d]=cross.transpose(-1,-2)
    return out


def contrast_moments(filtered,G,Q,left,right,left_weights,right_weights):
    """Two-time linear contrast without covariance subtraction.

    Backward conditional innovations are independent given the right state.
    Their quadratic forms sum to the contrast variance, avoiding cancellation
    in Var(x_right)+Var(x_left)-2 Cov(x_left,x_right).
    """
    if not 0<=left<right<filtered[1].shape[-2]:
        raise ValueError('Require ordered, distinct time indices')
    m,c,j=conditional_moments(filtered,G,Q)
    mean=(left_weights*m[...,left,:]).sum(-1)+(right_weights*m[...,right,:]).sum(-1)
    weight=left_weights.expand_as(m[...,left,:]);variance=torch.zeros_like(mean)
    eye=torch.eye(G.shape[-1],dtype=G.dtype,device=G.device)
    for t in range(left,right):
        gain=j[...,t,:,:];A=eye-gain@G
        conditional=A@filtered[2][...,t,:,:]@A.transpose(-1,-2)+gain@Q@gain.transpose(-1,-2)
        variance=variance+(weight*(conditional@weight.unsqueeze(-1)).squeeze(-1)).sum(-1)
        weight=(gain.transpose(-1,-2)@weight.unsqueeze(-1)).squeeze(-1)
    weight=weight+right_weights
    variance=variance+(weight*(c[...,right,:,:]@weight.unsqueeze(-1)).squeeze(-1)).sum(-1)
    if not bool(torch.isfinite(variance).all() and (variance>=0).all()):
        raise FloatingPointError('Invalid conditional contrast variance; no clipping')
    return mean,variance
