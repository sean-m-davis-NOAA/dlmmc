"""Offline float64 simulation smoothing without unused covariance outputs.

The Durbin–Koopman simulation correction needs only the RTS conditional mean.
RNG calls, filtering and mean-operation order match tensor_smoothing exactly.
The full Joseph covariance smoother remains the numerical reference.
"""
import torch
from .precision_prototype import filter_tensor


def conditional_mean(filtered, G, Q):
    _, m, P = filtered
    means = [m[..., -1, :]]
    for t in range(m.shape[-2]-2, -1, -1):
        prediction = (G @ m[..., t, :, None]).squeeze(-1)
        R = G @ P[..., t, :, :] @ G.transpose(-1, -2)+Q
        J = torch.linalg.solve(R, G @ P[..., t, :, :]).transpose(-1, -2)
        means.append(m[..., t, :]+(J @ (means[-1]-prediction).unsqueeze(-1)).squeeze(-1))
    return torch.stack(means[::-1], -2)


def conditional_draw_mean(y, variance, observed, F, G, Q, C, *, generator):
    if generator is None:
        raise ValueError('Supply an independent reconstruction RNG generator')
    if G.dtype != torch.float64 or G.device.type != 'cpu':
        raise ValueError('Offline CPU float64 scope only')
    if any(x.dtype != G.dtype or x.device != G.device for x in (y,variance,F,Q,C)):
        raise ValueError('Mixed dtype/device is not supported')
    if observed.dtype != torch.bool or observed.device != G.device:
        raise ValueError('Boolean mask must reside on input device')
    if not bool(torch.isfinite(y[observed]).all() and torch.isfinite(variance[observed]).all() and (variance[observed]>=0).all()):
        raise ValueError('Invalid observed data/variance')
    diagonal = Q.diagonal(dim1=-2,dim2=-1)
    if not bool(torch.isfinite(Q).all() and (diagonal>=0).all() and (Q==torch.diag_embed(diagonal)).all()):
        raise ValueError('Simulation smoother requires diagonal PSD process noise')
    y = torch.where(observed,y,torch.zeros_like(y))
    variance = torch.where(observed,variance,torch.zeros_like(variance))
    n,d = F.shape[-2:]
    shape = torch.broadcast_shapes(G.shape[:-2],F.shape[:-2],y.shape[:-1])
    def normal(tail):
        return torch.randn((*shape,*tail),dtype=G.dtype,device=G.device,generator=generator)
    state = (torch.linalg.cholesky(C) @ normal((d,)).unsqueeze(-1)).squeeze(-1)
    process_root = torch.sqrt(diagonal)
    states,pseudo = [],[]
    for t in range(n):
        state = (G @ state.unsqueeze(-1)).squeeze(-1)+process_root*normal((d,))
        states.append(state)
        pseudo.append((F[...,t,:]*state).sum(-1)+torch.sqrt(variance[...,t])*normal(()))
    simulated = torch.stack(states,-2)
    residual = y-torch.stack(pseudo,-1)
    filtered = filter_tensor(residual,variance.expand_as(residual),observed.expand_as(residual),F,G,Q,C)
    return simulated+conditional_mean(filtered,G,Q)
