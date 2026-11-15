"""Stage-01 QR square-root filter experiment, not a production qualification.

Diagonal positive-semidefinite process noise is supported, including exact zero.
The initial covariance must be positive definite. No jitter or eigenvalue floors.
"""
import torch


def filter_square_root(y, variance, observed, F, G, Q, C):
    if any(x.dtype != G.dtype or x.device != G.device for x in (y,variance,F,Q,C)):
        raise ValueError('Mixed dtype/device is not supported')
    if observed.dtype != torch.bool or observed.device != G.device:
        raise ValueError('Boolean mask must reside on the input device')
    if not bool(torch.isfinite(y[observed]).all() and torch.isfinite(variance[observed]).all() and (variance[observed]>=0).all()):
        raise ValueError('Invalid observed data/variance')
    diagonal = Q.diagonal(dim1=-2,dim2=-1)
    if not bool(torch.isfinite(Q).all() and (diagonal>=0).all() and (Q==torch.diag_embed(diagonal)).all()):
        raise ValueError('This experiment supports only finite diagonal PSD process noise')
    y=torch.where(observed,y,torch.zeros_like(y))
    variance=torch.where(observed,variance,torch.ones_like(variance))
    n,d=F.shape[-2:]
    shape=torch.broadcast_shapes(G.shape[:-2],F.shape[:-2],y.shape[:-1])
    L=torch.linalg.cholesky(C.expand(*shape,d,d))
    noise=torch.diag_embed(torch.sqrt(diagonal)).expand(*shape,d,d)
    m=G.new_zeros((*shape,d));terms=[];means=[];covariances=[]
    for t in range(n):
        m=(G@m.unsqueeze(-1)).squeeze(-1)
        prediction=torch.cat((G@L,noise),dim=-1)
        L=torch.linalg.qr(prediction.transpose(-1,-2),mode='r').R.transpose(-1,-2)
        f=F[...,t,:]
        joint=G.new_zeros((*shape,d+1,d+1))
        joint[...,0,0]=torch.sqrt(variance[...,t])
        joint[...,0,1:]=(f.unsqueeze(-2)@L).squeeze(-2)
        joint[...,1:,1:]=L
        lower=torch.linalg.qr(joint.transpose(-1,-2),mode='r').R.transpose(-1,-2)
        innovation_root=lower[...,0,0]
        if not bool(torch.isfinite(innovation_root).all() and (innovation_root!=0).all()):
            raise ValueError(f'Invalid innovation variance at time {t}')
        residual=y[...,t]-(f*m).sum(-1)
        updated=m+lower[...,1:,0]*(residual/innovation_root).unsqueeze(-1)
        mask=observed[...,t]
        m=torch.where(mask[...,None],updated,m)
        L=torch.where(mask[...,None,None],lower[...,1:,1:],L)
        terms.append(torch.where(mask,-.5*(G.new_tensor(2*torch.pi).log()+2*innovation_root.abs().log()+(residual/innovation_root)**2),torch.zeros_like(residual)))
        means.append(m);covariances.append(L@L.transpose(-1,-2))
    return torch.stack(terms,-1),torch.stack(means,-2),torch.stack(covariances,-3)
