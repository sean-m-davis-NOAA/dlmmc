"""Immutable component specifications and a backend-independent state layout.

A serialized model is self-contained JSON, never a process-local registry key.
Preset strings are compatibility shorthands, not the space of allowed models.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import json
import math
import re

PREFIX = 'dlmmc-model-v1:'


class Component:
    def __add__(self, other):
        return Model(self) + other


@dataclass(frozen=True)
class Trend(Component):
    """Local linear level/slope; positive half-normal slope-innovation scale."""
    level_variance: float = 1e-10


@dataclass(frozen=True)
class Level(Component):
    """Random-walk level with a positive half-normal innovation scale."""
    pass


@dataclass(frozen=True)
class Harmonics(Component):
    """Rotation pairs; periods in months, one shared scale per named block."""
    periods: tuple = (12., 6.)
    name: str = 'seas'

    def __post_init__(self):
        object.__setattr__(self, 'periods', tuple(float(p) for p in self.periods))
        if not self.periods or any(not math.isfinite(p) or p <= 0 for p in self.periods):
            raise ValueError('Harmonic periods must be finite and positive')
        if len(set(self.periods)) != len(self.periods):
            raise ValueError('Duplicate harmonic periods are not identifiable')
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', self.name) or self.name in ('trend','level','AR','reg'):
            raise ValueError('Unique harmonic name required; trend/level/AR/reg are reserved')


@dataclass(frozen=True)
class Regression(Component):
    """All provided predictors; dynamic=True/all, False/none, or named subset.

    static_variance=None preserves the historic 1e-20 (AR2) /1e-10 policy.
    This fixed variance is part of the model and is never a numerical floor.
    """
    dynamic: bool | tuple = False
    static_variance: float | None = None

    def __post_init__(self):
        if not isinstance(self.dynamic, bool):
            if isinstance(self.dynamic, str):raise ValueError('Use a tuple of dynamic regressor names')
            object.__setattr__(self, 'dynamic', tuple(self.dynamic))
            if len(set(self.dynamic)) != len(self.dynamic) or not all(isinstance(x,str) for x in self.dynamic):
                raise ValueError('Unique dynamic regressor names required')


@dataclass(frozen=True)
class AR(Component):
    """AR(1) or AR(2), with the existing positive conditional-uniform prior."""
    order: int = 1
    auxiliary_variance: float = 1e-20

    def __post_init__(self):
        if type(self.order) is not int or self.order not in (1,2):
            raise ValueError('AR order must be 1 or 2; omit AR for no autoregression')


_TYPES = {c.__name__: c for c in (Trend, Level, Harmonics, Regression, AR)}


@dataclass(frozen=True, init=False)
class Model:
    """Compose supported blocks, e.g. Model(Trend(), Harmonics(), AR(2)).

    Order is canonical (level/trend, harmonics, regression, AR), independent of
    the order used to compose the expression. Distinct harmonic blocks may have
    distinct scales. No hidden model registration or executable serialization.
    """
    components: tuple

    def __init__(self, *components):
        if not components or any(type(c).__name__ not in _TYPES or type(c) is not _TYPES[type(c).__name__] for c in components):
            raise ValueError('Compose one or more supported component objects')
        for cls in (Regression, AR):
            if sum(isinstance(c,cls) for c in components)>1:raise ValueError(f'At most one {cls.__name__} block')
        if sum(isinstance(c,(Trend,Level)) for c in components)>1:raise ValueError('Choose at most one Trend or Level')
        harmonics=[c for c in components if isinstance(c,Harmonics)]
        if len({c.name for c in harmonics})!=len(harmonics):raise ValueError('Unique harmonic block names required')
        periods=[v for c in harmonics for v in c.periods]
        if len(set(periods))!=len(periods):raise ValueError('Harmonic periods must not overlap between blocks')
        for c in components:
            for attr in ('level_variance','static_variance','auxiliary_variance'):
                v=getattr(c,attr,None)
                if v is not None and (not math.isfinite(v) or v<0):raise ValueError('Fixed variances must be finite and nonnegative')
        rank={Trend:0,Level:0,Harmonics:1,Regression:2,AR:3}
        object.__setattr__(self,'components',tuple(sorted(components,key=lambda c:(rank[type(c)],getattr(c,'name','')))))

    def __add__(self, other):
        return Model(*self.components, *(other.components if isinstance(other,Model) else (other,)))

    def to_dict(self):
        return {'version':1,'components':[dict(type=type(c).__name__,**asdict(c)) for c in self.components]}

    def to_json(self):
        return PREFIX+json.dumps(self.to_dict(),sort_keys=True,separators=(',',':'))

    @classmethod
    def from_dict(cls, value):
        if set(value)!= {'version','components'} or value['version']!=1:raise ValueError('Unsupported component model schema')
        try:return cls(*[_TYPES[c['type']](**{k:v for k,v in c.items() if k!='type'}) for c in value['components']])
        except (KeyError,TypeError) as exc:raise ValueError('Invalid component model') from exc

    @classmethod
    def from_preset(cls, name):
        from .api import _MODEL_SPECS
        s=_MODEL_SPECS[name]
        return cls(Trend(), *([Harmonics()] if s['seasonal'] else []),
                   *([Regression(dynamic=s['regressors']=='dynamic')] if s['regressors']!='none' else []),
                   AR(2 if s['ar2'] else 1))

    def bind(self, regressor_names):
        """Resolve named dynamic predictors to indices for portable checkpoints."""
        names=tuple(regressor_names)
        if len(set(names))!=len(names):raise ValueError('Unique regressor names required')
        return layout(self.to_json(),len(names),names)


def model_key(model):
    if isinstance(model,dict):model=Model.from_dict(model)
    return model.to_json() if isinstance(model,Model) else model


@lru_cache(maxsize=256)
def as_model(key):
    if isinstance(key,Model):return key
    if not isinstance(key,str):raise ValueError('Expected a Model or preset name')
    if key.startswith(PREFIX):
        value=json.loads(key[len(PREFIX):]);value.pop('regressor_names',None)
        return Model.from_dict(value)
    return Model.from_preset(key)


@lru_cache(maxsize=256)
def component_spec(model):
    m=as_model(model_key(model));reg=next((c for c in m.components if isinstance(c,Regression)),None)
    return {'seasonal':any(isinstance(c,Harmonics) for c in m.components),
            'ar2':any(isinstance(c,AR) and c.order==2 for c in m.components),
            'regressors':'none' if reg is None else 'dynamic' if reg.dynamic is True else 'static',
            'component_model':model_key(model)}


@dataclass(frozen=True)
class Layout:
    dimension: int
    parameters: tuple
    scale_defaults: tuple
    diagonal_scales: tuple
    fixed_diagonal: tuple
    rotations: tuple
    regressor_start: int
    dynamic_indices: tuple
    ar_start: int
    ar_order: int
    level_kind: str | None
    slices: tuple
    transition_columns: tuple


@lru_cache(maxsize=512)
def layout(model,nreg,regressor_names=()):
    m=as_model(model);parameters=[];scales=[];diag=[];fixed=[];rotations=[];slices=[]
    d=0;regstart=-1;arstart=-1;arorder=0;level=None;dynamic=()
    ar2=any(isinstance(c,AR) and c.order==2 for c in m.components)
    def parameter(name,scale):
        parameters.append(name);scales.append((name,scale));return name
    for c in m.components:
        start=d
        if isinstance(c,Trend):
            level='trend';d+=2;fixed.append((start,c.level_variance));diag.append((start+1,parameter('sigma_trend',1e-4)))
        elif isinstance(c,Level):
            level='level';d+=1;diag.append((start,parameter('sigma_level',1e-4)))
        elif isinstance(c,Harmonics):
            name=parameter('sigma_'+c.name,.01)
            for period in c.periods:
                rotations.append((d,period));diag.extend(((d,name),(d+1,name)));d+=2
        elif isinstance(c,Regression):
            if nreg<0:raise ValueError('Nonnegative regressor count required')
            regstart=d;d+=nreg
            if c.dynamic is True:dynamic=tuple(range(nreg))
            elif c.dynamic is False:dynamic=()
            else:
                # Bound keys carry stable numeric indices; named inputs bind in prepare_data.
                if all(isinstance(v,int) for v in c.dynamic):dynamic=tuple(c.dynamic)
                else:
                    if not regressor_names or any(v not in regressor_names for v in c.dynamic):raise ValueError('Dynamic regressor names must match supplied predictors')
                    dynamic=tuple(i for i,n in enumerate(regressor_names) if n in c.dynamic)
            for j in range(nreg):
                if j in dynamic:diag.append((start+j,'sigma_reg[%d]'%j))
                else:fixed.append((start+j,((1e-20 if ar2 else 1e-10) if c.static_variance is None else c.static_variance)))
        elif isinstance(c,AR):
            arstart=d;arorder=c.order;d+=arorder;diag.append((start,parameter('sigma_AR',.5)))
            if arorder==2:fixed.append((start+1,c.auxiliary_variance))
        slices.append((type(c).__name__ if not isinstance(c,Harmonics) else c.name,start,d))
    if nreg and regstart<0:raise ValueError('Model without Regression requires zero predictors')
    # Preserve the legacy transformed order: scales then regressor scales then AR.
    for j in dynamic:parameter('sigma_reg[%d]'%j,1e-4)
    if arorder:parameters.append('rhoAR1')
    if arorder==2:parameters.append('rhoAR2')
    c0=list(range(d));c1=list(range(d))
    if level=='trend':c1[0]=1
    for i,_ in rotations:c0[i]=c0[i+1]=i;c1[i]=c1[i+1]=i+1
    if arorder==2:c0[arstart]=c0[arstart+1]=arstart;c1[arstart]=c1[arstart+1]=arstart+1
    return Layout(d,tuple(parameters),tuple(scales),tuple(diag),tuple(fixed),tuple(rotations),regstart,dynamic,arstart,arorder,level,tuple(slices),(tuple(c0),tuple(c1)))


def bound_model(model,names):
    """Persist named selections and predictor order in a self-contained key."""
    key=model_key(model)
    if not isinstance(key,str) or not key.startswith(PREFIX):return key
    m=as_model(key);l=layout(key,len(names),tuple(names))
    # Keep public named specs intact; store binding alongside schema in the key.
    value=m.to_dict();value['regressor_names']=list(names)
    return PREFIX+json.dumps(value,sort_keys=True,separators=(',',':'))


@lru_cache(maxsize=512)
def compiled_layout(model,nreg):
    key=model_key(model)
    if isinstance(key,str) and key.startswith(PREFIX):
        value=json.loads(key[len(PREFIX):]);names=tuple(value.pop('regressor_names',()))
        if names and len(names)!=nreg:raise ValueError('Bound predictor count mismatch')
        key=Model.from_dict(value).to_json()
        return layout(key,nreg,names)
    return layout(key,nreg)


def defaults(model,nreg):
    return dict(compiled_layout(model,nreg).scale_defaults)


def resolve_priors(model,nreg,priors=None):
    l=compiled_layout(model,nreg);result={'S':10.}
    for name,scale in l.scale_defaults:
        result['sigma_reg_prior' if name.startswith('sigma_reg[') else name+'_prior']=scale
    unknown=set(priors or {})-set(result)
    if unknown:raise ValueError('Priors not used by this model: '+str(sorted(unknown)))
    result.update(priors or {})
    import numpy as np
    for key,v in result.items():
        a=np.asarray(v,dtype=float)
        if np.any(~np.isfinite(a)) or np.any(a<=0):raise ValueError('Positive finite prior scales required')
        if key=='sigma_reg_prior':
            if a.ndim==0:a=np.full(nreg,a)
            if a.shape!=(nreg,):raise ValueError('sigma_reg_prior must match predictor count')
            result[key]=a.tolist()
        elif a.ndim:raise ValueError('Scalar prior scale required')
        else:result[key]=float(a)
    return result


def regressor_count(model,dimension):
    m=as_model(model_key(model))
    base=sum(2 if isinstance(c,Trend) else 1 if isinstance(c,Level) else 2*len(c.periods) if isinstance(c,Harmonics) else c.order if isinstance(c,AR) else 0 for c in m.components)
    k=dimension-base
    if k<0 or (not any(isinstance(c,Regression) for c in m.components) and k):raise ValueError('State dimension does not match components')
    compiled_layout(model,k)
    return k


def require_level(model,nreg):
    if compiled_layout(model,nreg).level_kind is None:
        raise ValueError('Trend/level summaries require a Trend or Level component; use state-space outputs for other models')


def initial_coordinates(model,nreg):
    import numpy as np
    return np.array([-9.2 if n in ('sigma_trend','sigma_level') or n.startswith('sigma_reg[') else -2.3 if n=='sigma_AR' else -4.6 if n.startswith('sigma_') else 0. for n in compiled_layout(model,nreg).parameters])


def build_state_space(model,parameters,regressors,*,priors=None,sampling=1.,regressor_names=None):
    """Return native Torch F/G/Q/C and transformed prior for explicit parameters.

    Inputs are Torch tensors on the chosen CPU/MPS/CUDA device with the same
    dtype. Final axes: parameters [P], regressors [time,K]; leading axes
    broadcast. Use bound_model/prepare_data to resolve named dynamic regressors.
    No filtering, inference, implicit precision conversion or sampled states.
    """
    from .tensor_target import model_components
    import torch
    key=model_key(model) if regressor_names is None else bound_model(model,regressor_names)
    k=regressors.shape[-1];l=compiled_layout(key,k)
    resolved=resolve_priors(key,k,priors)
    y=regressors.new_zeros(regressors.shape[:-1]);observed=torch.zeros_like(y,dtype=torch.bool)
    F,G,Q,C,lp,valid=model_components(parameters,y,y,observed,regressors,key,S=resolved['S'],sampling=sampling)
    for j,name in enumerate(l.parameters):
        if not name.startswith('sigma'):continue
        old=dict(l.scale_defaults)[name];new=resolved['sigma_reg_prior'][int(name.split('[')[1][:-1])] if name.startswith('sigma_reg[') else resolved[name+'_prior']
        if old!=new:
            theta=torch.exp(torch.where(valid,parameters[...,j],torch.zeros_like(parameters[...,j])))
            lp=lp+math.log(old/new)+.5*(theta/old)**2-.5*(theta/new)**2
    return dict(observation=F,transition=G,innovation=Q,initial=C,log_prior=lp,valid=valid,
                parameter_names=list(l.parameters),component_slices={name:slice(a,b) for name,a,b in l.slices})

