"""Fixed-coverage monthly ENSO(t-lag), with explicit native precision.

The prior is Gaussian directly on a bounded positive lag field. Logistic
sampling coordinates carry their Jacobian; neither data windows nor predictor
coverage depend on proposals. Other regressors retain their fixed inputs.
"""
from dataclasses import dataclass,field
import math
import numpy as np
import torch
from .spatial import GaussianSpatialPrior
from .tensor_target import model_components
from .torch_filter import TorchTarget,log_likelihood


@dataclass(frozen=True)
class MonthlyENSO:
    """Finite regular source and fixed observations in relative month units.

    Source values are [source time] or [cell,source time]. Missing source values
    or gaps are rejected, never imputed. ``upper_lag`` specifies required record
    coverage; the sampling prior separately enforces strict positive support.
    """
    source_month: object
    values: object
    observation_month: object
    upper_lag: float=12.
    observation_anchor_ns: int | None=None
    origin_month: float=field(init=False)

    def __post_init__(self):
        t=np.asarray(self.source_month,dtype=float).copy()
        y=np.asarray(self.values,dtype=float).copy()
        obs=np.asarray(self.observation_month,dtype=float).copy()
        if t.ndim!=1 or len(t)<2 or obs.ndim!=1 or not len(obs):raise ValueError('One-dimensional nonempty time axes required')
        if y.ndim not in (1,2) or y.shape[-1]!=len(t):raise ValueError('Source values must end in source-time axis')
        if not all(np.isfinite(x).all() for x in (t,y,obs)):raise ValueError('Finite source/observation times and complete predictor required')
        if not np.all(np.diff(t)==1) or not np.all(np.diff(obs)>0):raise ValueError('Regular monthly source without gaps and increasing observations required')
        if not np.isfinite(self.upper_lag) or self.upper_lag<=0:raise ValueError('Positive finite upper lag and fixed coverage required')
        if obs.min()-self.upper_lag<t[0] or obs.max()>t[-1]:raise ValueError('Predictor does not cover the fixed observation window at all allowed lags')
        # Subtract one shared origin before dtype conversion; large absolute
        # date numbers never enter float32 interpolation arithmetic.
        object.__setattr__(self,'origin_month',float(t[0]))
        obs=obs-t[0];t=t-t[0]
        for name,array in [('source_month',t),('values',y),('observation_month',obs)]:
            array.setflags(write=False);object.__setattr__(self,name,array)

    @classmethod
    def from_dates(cls,source_dates,values,observation_dates,*,upper_lag=12.):
        source=np.asarray(source_dates,dtype='datetime64[ns]');obs=np.asarray(observation_dates,dtype='datetime64[ns]')
        if np.isnat(source).any() or np.isnat(obs).any():raise ValueError('Finite monthly dates required')
        sm=source.astype('datetime64[M]');om=obs.astype('datetime64[M]')
        anchors=np.r_[source-sm.astype('datetime64[ns]'),obs-om.astype('datetime64[ns]')]
        if not np.all(anchors==anchors[0]):raise ValueError('Source and observations need the same within-month anchor')
        return cls(sm.astype(np.int64),values,om.astype(np.int64),upper_lag,int(anchors[0].astype('timedelta64[ns]').astype(np.int64)))

    def bind(self,*,cells,dtype,device):
        return BoundMonthlyENSO(self,cells=cells,dtype=dtype,device=device)


class BoundMonthlyENSO:
    def __init__(self,spec,*,cells,dtype,device):
        if dtype not in (torch.float32,torch.float64) or cells<1:raise ValueError('Explicit floating dtype and positive cells required')
        values=spec.values
        if values.ndim==1:values=np.broadcast_to(values,(cells,len(values)))
        if values.shape[0]!=cells:raise ValueError('Source cell axis differs from prepared panels')
        self.values=torch.tensor(values.copy(),dtype=dtype,device=device)
        self.time=torch.tensor(spec.observation_month.copy(),dtype=dtype,device=device)
        self.spec=spec;self.cells=cells
        if not bool(torch.isfinite(self.values).all()):raise ValueError('Predictor overflows requested dtype')

    def __call__(self,lag):
        """Return [walker,cell,time]; fixed zero is allowed for reference tests."""
        if lag.ndim!=2 or lag.shape[1]!=self.cells or lag.dtype!=self.values.dtype or lag.device!=self.values.device:raise ValueError('Lag shape/dtype/device mismatch')
        if not bool(torch.isfinite(lag).all() and (lag>=0).all() and (lag<=self.spec.upper_lag).all()):raise ValueError('Lag outside fixed coverage support')
        query=self.time[None,None,:]-lag[...,None]
        n=self.values.shape[-1]
        if not bool(((query>=0)&(query<=n-1)).all()):raise ValueError('No boundary extrapolation or clamping permitted')
        left=torch.floor(query).to(torch.long)
        # At the exact final source timestamp, use the last interval with
        # weight one. This is endpoint interpolation, not out-of-range clamping.
        left=torch.where(left==n-1,left-1,left)
        weight=query-left.to(query.dtype)
        values=self.values[None].expand(lag.shape[0],-1,-1)
        low=torch.gather(values,2,left);high=torch.gather(values,2,left+1)
        return low+weight*(high-low)


class ENSOLagTarget:
    """Two prior-independent Gaussian fields plus conditionally local nuisance.

    Flat layout per cell is existing model coordinates followed by unconstrained
    ENSO lag. Sigma and lag use identical spatial correlation lengths/coordinates
    but their own fixed means/scales/units. Both orthant/bounded normalizers are
    constants while these prior hyperparameters remain fixed.
    """
    def __init__(self,prepared,sigma_prior,lag_prior,predictor,*,backend='numba'):
        p=prepared
        if p.regressor_names.count('ENSO')!=1:raise ValueError('Exactly one explicitly named ENSO regressor required')
        if not isinstance(predictor,MonthlyENSO):raise ValueError('A fixed-coverage MonthlyENSO is required')
        if len(predictor.observation_month)!=len(p.time):raise ValueError('Predictor and observations have different time counts')
        times=np.asarray(p.time)
        if np.issubdtype(times.dtype,np.datetime64):
            months=times.astype('datetime64[M]')
            if predictor.observation_anchor_ns is None:raise ValueError('Use MonthlyENSO.from_dates for datetime observations')
            anchors=(times.astype('datetime64[ns]')-months.astype('datetime64[ns]')).astype(np.int64)
            if not np.all(anchors==predictor.observation_anchor_ns):raise ValueError('Observation monthly anchor differs from predictor specification')
            times=months.astype(np.int64)
        if not np.array_equal(times-predictor.origin_month,predictor.observation_month):raise ValueError('Predictor observation timestamps differ from prepared data')
        if backend not in ('numba','torch','mps','cuda'):raise ValueError('Explicit numba, torch, mps or cuda backend required')
        if backend in ('mps','cuda') and (p.y.device.type!=backend or (backend=='mps' and p.y.dtype!=torch.float32)):raise ValueError('Native GPU prepared inputs required; MPS uses float32')
        if backend=='numba' and (p.y.device.type!='cpu' or p.y.dtype not in (torch.float32,torch.float64)):raise ValueError('Numba requires CPU32/64')
        if sigma_prior.lower!=0 or not math.isinf(sigma_prior.upper):raise ValueError('Sigma prior requires strict positive unbounded support')
        if lag_prior.lower!=0 or not np.isfinite(lag_prior.upper) or lag_prior.upper!=predictor.upper_lag:raise ValueError('Lag prior support and predictor coverage must match')
        if lag_prior.field_unit!='months':raise ValueError('Lag prior units must be months')
        for name in ('vertical','latitude','length_vertical','length_latitude','vertical_coordinate','vertical_unit','latitude_unit'):
            if getattr(sigma_prior,name)!=getattr(lag_prior,name):raise ValueError('Both fields must share spatial kernel coordinates/lengths/units')
        if len(sigma_prior.mean)!=p.cells or len(lag_prior.mean)!=p.cells:raise ValueError('One field element per panel required')
        self.prepared=p;self.sigma_prior=sigma_prior;self.lag_prior=lag_prior;self.predictor=predictor;self.backend=backend
        self.base_parameters=len(p.parameter_names);self.parameters=self.base_parameters+1;self.dimension=p.cells*self.parameters
        self.sigma_index=p.parameter_names.index('sigma_trend');self.lag_index=self.base_parameters;self.enso_index=p.regressor_names.index('ENSO')
        self.field_indices=(self.sigma_index,self.lag_index)
        self.fields=(sigma_prior.bind(dtype=p.y.dtype,device=p.y.device),lag_prior.bind(dtype=p.y.dtype,device=p.y.device))
        self.interpolator=predictor.bind(cells=p.cells,dtype=p.y.dtype,device=p.y.device);self.correction=TorchTarget(p)

    def lag_values(self,u):
        v=u[...,self.lag_index];prob=torch.sigmoid(v);lag=self.lag_prior.upper*prob
        valid=torch.isfinite(v)&(lag>0)&(lag<self.lag_prior.upper)
        jac=math.log(self.lag_prior.upper)-torch.nn.functional.softplus(v)-torch.nn.functional.softplus(-v)
        return lag,jac,valid

    def regressors(self,u):
        lag,_,valid=self.lag_values(u)
        safe=torch.where(valid,lag,torch.full_like(lag,self.lag_prior.upper/2))
        x=self.prepared.regressors[None].expand(u.shape[0],-1,-1,-1).clone()
        x[...,self.enso_index]=self.interpolator(safe)
        return x

    def local(self,u):
        if u.ndim!=3 or u.shape[1:]!=(self.prepared.cells,self.parameters):raise ValueError('Expected [walker,cell,base parameters+lag]')
        p=self.prepared;base=u[...,:self.base_parameters];_,jac,valid_lag=self.lag_values(u);x=self.regressors(u)
        if self.backend=='numba':
            from .numba_prototype import target_numba
            value=target_numba(base,p.y,p.variance,p.observed,x,p.model,S=p.priors['S'],sampling=p.sampling,algorithm='structured',research_float32=True)
            value=value+self.correction.prior_correction(base,torch.isfinite(value))
        else:
            F,G,Q,C,lp,valid=model_components(base,p.y,p.variance,p.observed,x,p.model,S=p.priors['S'],sampling=p.sampling)
            if self.backend=='mps':
                from .structured_mps import filter_mps
                ll=filter_mps(p.y.contiguous(),p.variance.contiguous(),p.observed.contiguous(),F.contiguous(),G,Q,C,model=p.model)
            else:ll=log_likelihood(p.y,p.variance,p.observed,F,G,Q,C)
            value=ll+lp+self.correction.prior_correction(base,valid)
            value=torch.where(valid,value,torch.full_like(value,-torch.inf))
        result=value+jac
        supported=torch.isfinite(value)&valid_lag
        if bool((supported&~torch.isfinite(result)).any()):raise FloatingPointError('Nonfinite supported lag-local density')
        return torch.where(supported,result,torch.full_like(result,-torch.inf))

    def joint(self,flat,local):
        u=flat.reshape(-1,self.prepared.cells,self.parameters);valid=torch.isfinite(local).all(-1)
        sigma=torch.where(valid[:,None],u[...,self.sigma_index],torch.zeros_like(u[...,self.sigma_index])).exp()
        lag,_,_=self.lag_values(u);lag=torch.where(valid[:,None],lag,torch.full_like(lag,self.lag_prior.upper/2))
        scale=float(self.prepared.priors['sigma_trend_prior'])
        old=.5*math.log(2/math.pi)-math.log(scale)-.5*(sigma/scale)**2
        value=local.sum(-1)-old.sum(-1)+self.fields[0].log_kernel(sigma)+self.fields[1].log_kernel(lag)
        if bool((valid&~torch.isfinite(value)).any()):raise FloatingPointError('Nonfinite supported joint lag target')
        return torch.where(valid,value,torch.full_like(value,-torch.inf))

    def __call__(self,flat):
        if flat.ndim!=2 or flat.shape[-1]!=self.dimension:raise ValueError('Lag target requires [walker,cell*(base parameters+lag)]')
        return self.joint(flat,self.local(flat.reshape(-1,self.prepared.cells,self.parameters)))
