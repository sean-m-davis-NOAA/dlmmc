"""Joint lag diagnostics preserve independent runs and coupled cell alignment."""
import itertools
import numpy as np
from scipy.special import expit
from .inference import physical_parameters,window_diagnostics,posterior_fingerprint
from .validation_derived import finite_moment_summaries,PROFILE,raw_variance_diagnostics


def lag_summaries(fit):
    u=fit.posterior().unconstrained.values;p=fit.prepared
    names,physical=physical_parameters(u[...,:-1],p.model,p.regressors.shape[-1]);names,physical=finite_moment_summaries(names,physical)
    lag=fit.config['lag_prior']['upper']*expit(u[...,-1]);physical=np.concatenate([physical,lag[...,None]],axis=-1);names=names+['lag_ENSO_months']
    labels=[f'cell{g}:{n}' for g in range(p.cells) for n in names];arrays=[physical.reshape(*physical.shape[:3],-1)]
    sigma=np.exp(u[...,p.parameter_names.index('sigma_trend')])/np.asarray(fit.config['sigma_prior']['scale'])
    normalized_lag=lag/np.asarray(fit.config['lag_prior']['scale'])
    for name,field in [('sigma',sigma),('lag',normalized_lag)]:
        arrays.append(field.mean(-1)[...,None]);labels.append(name+'_field_mean')
        for i,j in itertools.combinations(range(p.cells),2):
            arrays.extend([(field[...,i]-field[...,j])[...,None],(field[...,i]*field[...,j])[...,None]])
            labels.extend([f'{name}_contrast:{i}-{j}',f'{name}_product:{i},{j}'])
    for i in range(p.cells):
        arrays.append((sigma[...,i]*normalized_lag[...,i])[...,None]);labels.append(f'sigma_lag_product:{i}')
    return labels,np.concatenate(arrays,-1),names,physical


def diagnose_enso_lag(fit,*,trace_review=None):
    posterior=fit.posterior();labels,values,names,physical=lag_summaries(fit);cells=[]
    _,raw=physical_parameters(posterior.unconstrained.values[...,:-1],fit.prepared.model,fit.prepared.regressors.shape[-1])
    for g in range(fit.prepared.cells):
        d=window_diagnostics(physical[...,g,:]);d.update(cell=g,coordinates={k:str(v[g]) for k,v in fit.prepared.cell_coordinates.items()},raw_stationary_AR_variance=raw_variance_diagnostics(raw[...,g,-1]));cells.append(d)
    joint=window_diagnostics(values);reviewed=isinstance(trace_review,str) and bool(trace_review.strip());eligible=reviewed and joint['eligible'] and all(c['eligible'] for c in cells)
    return dict(schema=1,profile=PROFILE,eligible=eligible,status='JOINT_LAG_PARAMETER_WINDOW_ELIGIBLE' if eligible else 'INCONCLUSIVE',trace_review=trace_review,posterior_sha256=posterior_fingerprint(fit,posterior),quantities=names,cells=cells,joint=joint,joint_quantities=labels,walkers_are_independent_chains=False,spatial_cells_are_independent=False,scope='Two fields/marginals/contrasts/products only; reconstructed scientific outputs need their own checks')
