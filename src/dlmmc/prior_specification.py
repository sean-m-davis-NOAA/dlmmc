"""Public prior widths and readable summaries; legacy names stay internal/compatible."""
from collections.abc import Mapping
import numpy as np

INITIAL_STATE_PRIOR_WIDTH = 3.0
INITIAL_STATE_PRIOR_VARIANCE = INITIAL_STATE_PRIOR_WIDTH**2


def canonical_priors(priors):
    """Translate public standard deviations to existing backend keys.

    Legacy S remains a variance. All sigma aliases are already standard
    deviations. Reject duplicate aliases, even if their values agree.
    """
    if priors is None:return {}
    if not isinstance(priors,Mapping):raise TypeError('priors must be a mapping')
    result={}
    for name,value in priors.items():
        key=name
        if name=='initial_state_prior_width':
            a=np.asarray(value,dtype=float)
            if a.ndim or not np.isfinite(a) or a<=0:raise ValueError('initial_state_prior_width must be a positive finite scalar standard deviation')
            with np.errstate(over='ignore',under='ignore'):value=float(a*a)
            if not np.isfinite(value) or value<=0:raise ValueError('initial_state_prior_width squared must be finite and positive; no floor')
            key='S'
        elif isinstance(name,str) and name.startswith('sigma_') and name.endswith('_prior_width'):
            key=name.removesuffix('_width')
        if key in result:raise ValueError(f'Duplicate prior aliases for {key}; supply only one spelling')
        result[key]=value
    return result


def combine_priors(priors,direct):
    result=canonical_priors(priors)
    for key,value in canonical_priors({k:v for k,v in direct.items() if v is not None}).items():
        if key in result:raise ValueError(f'Set {key!r} either directly or in priors, not both')
        result[key]=value
    return result


def _number(value):
    a=np.asarray(value)
    return f'{float(a):.7g}' if a.ndim==0 else '['+', '.join(f'{x:.7g}' for x in a.ravel())+']'


def announce_model(model,regressor_names,priors,*,panels=1,fields=None):
    """Print the effective model/prior, including any replacing spatial fields."""
    from .components import compiled_layout,as_model,Trend,Level,Harmonics,Regression,AR
    layout=compiled_layout(model,len(regressor_names));pieces=[]
    for c in as_model(model).components:
        if isinstance(c,Trend):pieces.append('Trend')
        elif isinstance(c,Level):pieces.append('Level')
        elif isinstance(c,Harmonics):pieces.append(f'Harmonics({c.name}: '+', '.join(f'{p:g}' for p in c.periods)+' months)')
        elif isinstance(c,Regression):
            dynamic=[regressor_names[i] for i in layout.dynamic_indices]
            pieces.append('Regression('+', '.join(regressor_names)+'; '+('dynamic: '+', '.join(dynamic) if dynamic else 'static')+')')
        elif isinstance(c,AR):pieces.append(f'AR({c.order})')
    print(f"DLMMC model ({panels} panel{'s' if panels!=1 else ''}): "+' + '.join(pieces),flush=True)
    print('  initial_state_prior_width='+_number(np.sqrt(priors['S']))+
          ' (Gaussian SD; initial variance='+_number(priors['S'])+'; AR state uses stationary covariance)',flush=True)
    fields=fields or {};printed=set()
    for name in layout.parameters:
        if not name.startswith('sigma') or name in fields:continue
        key='sigma_reg_prior' if name.startswith('sigma_reg[') else name+'_prior'
        if key not in printed:
            print(f'  {key}_width={_number(priors[key])}: HalfNormal scale (Gaussian SD before positive truncation)',flush=True)
            printed.add(key)
    if layout.ar_order==1:print('  rhoAR1 ~ Uniform(0, 1)',flush=True)
    elif layout.ar_order==2:print('  rhoAR1 ~ Uniform(0, 0.5); rhoAR2 | rhoAR1 ~ Uniform(0, 1-rhoAR1)',flush=True)
    for name,f in fields.items():
        print(f'  {name}: truncated Gaussian field, mean={_number(f.mean)}, std={_number(f.scale)}, '
              f'bounds=({f.lower:g}, {f.upper:g}), lengths=({f.length_vertical:g} {f.vertical_unit}, {f.length_latitude:g} {f.latitude_unit}), units={f.field_unit}',flush=True)
    if len(fields)>1:print('  Spatial fields are prior-independent.',flush=True)
    print('  Model scales use normalized response units; responses are mean/range normalized internally; predictors retain supplied units.',flush=True)
