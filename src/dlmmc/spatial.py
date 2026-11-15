"""Fixed Gaussian field priors in explicitly named spatial coordinates.

Only sigma_trend is spatially coupled. Truncation normalizers are constant
for these fixed hyperparameters; log kernels omit that constant explicitly.
No numerical nugget, clipping, or log-Gaussian replacement is used.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch


def test_lengths_from_correlation(vertical_spacing, latitude_spacing, rho=.5):
    """Test fixture helper; production construction accepts lengths directly."""
    if not 0 < rho < 1 or min(vertical_spacing, latitude_spacing) <= 0:
        raise ValueError('Positive spacings and 0 < rho < 1 required')
    divisor=math.sqrt(-2*math.log(rho))
    return vertical_spacing/divisor,latitude_spacing/divisor


@dataclass(frozen=True)
class GaussianSpatialPrior:
    """Gaussian on physical field values, jointly truncated to fixed bounds.

    ``vertical_coordinate`` must explicitly identify altitude or log_pressure.
    Log-pressure means ln(p/p_ref); differences are independent of p_ref.
    Field mean/scale units must match the modeled (possibly normalized) field.
    Changing lengths or hyperparameters requires a new target/checkpoint.
    """
    vertical: tuple
    latitude: tuple
    length_vertical: float
    length_latitude: float
    vertical_coordinate: str
    vertical_unit: str
    latitude_unit: str
    mean: tuple
    scale: tuple
    field_unit: str
    lower: float = 0.
    upper: float = math.inf

    def __post_init__(self):
        for name in ('vertical','latitude','mean','scale'):
            object.__setattr__(self,name,tuple(float(v) for v in getattr(self,name)))
        n=len(self.vertical)
        if not n or any(len(getattr(self,k))!=n for k in ('latitude','mean','scale')):
            raise ValueError('One coordinate, mean and scale per field element required')
        if not all(np.isfinite(getattr(self,k)).all() for k in ('vertical','latitude','mean','scale')):
            raise ValueError('Finite coordinates and hyperparameters required')
        if not np.isfinite([self.length_vertical,self.length_latitude]).all() or min(self.length_vertical,self.length_latitude)<=0 or min(self.scale)<=0:
            raise ValueError('Positive finite scales and lengths required')
        if self.vertical_coordinate not in ('altitude','log_pressure'):
            raise ValueError('Explicit altitude or log_pressure coordinate required')
        if not self.vertical_unit or not self.latitude_unit or not self.field_unit:
            raise ValueError('Coordinate and field units required')
        if self.vertical_coordinate=='log_pressure' and self.vertical_unit!='dimensionless_log_pressure_ratio':
            raise ValueError('Log-pressure must be labeled dimensionless_log_pressure_ratio')
        if not self.lower < self.upper or math.isnan(self.lower) or math.isnan(self.upper):
            raise ValueError('Strict ordered bounds required')
        if len(set(zip(self.vertical,self.latitude)))!=n:
            raise ValueError('Duplicate coordinates make the Gaussian covariance singular')
        # Fail explicitly if the requested covariance is not numerically SPD.
        np.linalg.cholesky(self.correlation())

    def correlation(self):
        z=np.asarray(self.vertical);lat=np.asarray(self.latitude)
        return np.exp(-.5*((z[:,None]-z)/self.length_vertical)**2
                      -.5*((lat[:,None]-lat)/self.length_latitude)**2)

    def covariance(self):
        scale=np.asarray(self.scale)
        return scale[:,None]*self.correlation()*scale[None,:]

    def bind(self, *, dtype=torch.float64, device='cpu'):
        return BoundGaussianField(self,dtype=dtype,device=device)


class BoundGaussianField:
    def __init__(self,prior,*,dtype,device):
        if dtype not in (torch.float32,torch.float64):raise ValueError('Explicit float32 or float64 required')
        self.prior=prior
        self.mean=torch.tensor(prior.mean,dtype=dtype,device=device)
        self.scale=torch.tensor(prior.scale,dtype=dtype,device=device)
        self.cholesky=torch.linalg.cholesky(torch.tensor(prior.correlation(),dtype=dtype,device=device))
        self.log_gaussian_constant=-self.scale.log().sum()-self.cholesky.diagonal().log().sum()-.5*len(prior.mean)*math.log(2*math.pi)

    def log_kernel(self,value):
        """Gaussian density inside strict support, without truncation normalizer."""
        if value.shape[-1]!=len(self.prior.mean) or value.dtype!=self.mean.dtype or value.device!=self.mean.device:
            raise ValueError('Field shape/dtype/device mismatch')
        valid=torch.isfinite(value).all(-1)&(value>self.prior.lower).all(-1)&(value<self.prior.upper).all(-1)
        safe=torch.where(valid[...,None],value,self.mean)
        z=(safe-self.mean)/self.scale
        whitened=torch.linalg.solve_triangular(self.cholesky,z.reshape(-1,len(self.mean)).T,upper=False).T.reshape(z.shape)
        result=self.log_gaussian_constant-.5*whitened.square().sum(-1)
        if bool((valid&~torch.isfinite(result)).any()):raise FloatingPointError('Nonfinite supported spatial density')
        return torch.where(valid,result,torch.full_like(result,-torch.inf))


class SpatialTarget:
    """One fully joint target; complementary-half acceptance uses G*P dimensions.

    Old sigma_trend half-normal terms are replaced exactly. Its log-coordinate
    Jacobian and every other local prior remain in the independent target.
    The fixed global truncation normalizer is omitted, so compare density
    differences (or supply that constant separately), not normalized evidence.
    """
    def __init__(self,prepared,prior,*,backend='numba'):
        from .sampling import make_target
        if len(prior.mean)!=prepared.cells:raise ValueError('Prior and prepared panels differ')
        if prior.lower!=0 or not math.isinf(prior.upper):raise ValueError('sigma_trend requires strict positive, unbounded support')
        self.prepared=prepared;self.prior=prior;self.local=make_target(prepared,backend)
        self.field=prior.bind(dtype=prepared.y.dtype,device=prepared.y.device)
        self.parameters=len(prepared.parameter_names);self.dimension=prepared.cells*self.parameters
        self.index=prepared.parameter_names.index('sigma_trend')
        self.old_scale=float(prepared.priors['sigma_trend_prior'])

    def __call__(self,flat):
        if flat.ndim!=2 or flat.shape[-1]!=self.dimension:raise ValueError('Joint sampler requires [walker,cell*parameter]')
        u=flat.reshape(-1,self.prepared.cells,self.parameters)
        local=self.local(u);valid=torch.isfinite(local).all(-1)
        sigma=torch.exp(torch.where(valid[:,None],u[...,self.index],torch.zeros_like(u[...,self.index])))
        old=.5*math.log(2/math.pi)-math.log(self.old_scale)-.5*(sigma/self.old_scale)**2
        result=local.sum(-1)-old.sum(-1)+self.field.log_kernel(sigma)
        if bool((valid&~torch.isfinite(result)).any()):raise FloatingPointError('Nonfinite supported joint target')
        return torch.where(valid,result,torch.full_like(result,-torch.inf))
