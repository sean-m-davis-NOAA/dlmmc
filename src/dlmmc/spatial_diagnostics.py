"""Joint-cell diagnostics keep ensemble replication and field alignment intact."""
import itertools
import numpy as np
from .inference import physical_parameters,window_diagnostics,diagnose_fit
from .validation_derived import finite_moment_summaries


def spatial_summaries(fit):
    """Panel marginals, standardized sigma contrasts and cross-products.

    Products diagnose covariance information in addition to marginal mixing.
    Cross-cell correlations require the same aligned joint draws; cells never
    supply extra chains. Scaling uses fixed prior scales, not fitted moments.
    """
    u=fit.posterior().unconstrained.values;p=fit.prepared
    names,values=physical_parameters(u,p.model,p.regressors.shape[-1]);names,values=finite_moment_summaries(names,values)
    labels=[f'cell{g}:{name}' for g in range(p.cells) for name in names]
    output=[values.reshape(*values.shape[:3],-1)]
    scale=np.asarray(fit.config['prior']['scale']);sigma=np.exp(u[...,p.parameter_names.index('sigma_trend')])/scale
    contrasts=[];products=[]
    for i,j in itertools.combinations(range(p.cells),2):
        contrasts.append(sigma[...,i]-sigma[...,j]);products.append(sigma[...,i]*sigma[...,j])
        labels.append(f'sigma_contrast:{i}-{j}')
    labels += [f'sigma_product:{i},{j}' for i,j in itertools.combinations(range(p.cells),2)]
    if contrasts:output.extend([np.stack(contrasts,-1),np.stack(products,-1)])
    return labels,np.concatenate(output,-1)


def diagnose_spatial(fit,*,trace_review=None):
    """Practical parameter and joint covariance-summary gate, not equivalence."""
    local=diagnose_fit(fit,trace_review=trace_review);names,values=spatial_summaries(fit)
    joint=window_diagnostics(values)
    local.update(joint=joint,joint_quantities=names,eligible=local['eligible'] and joint['eligible'],
        status='JOINT_PARAMETER_WINDOW_ELIGIBLE' if local['eligible'] and joint['eligible'] else 'INCONCLUSIVE',
        scope='Joint marginals, sigma contrasts and cross-products; reconstructed-output and cross-reference checks remain separate',spatial_cells_are_independent=False)
    return local
