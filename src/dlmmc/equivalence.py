"""Offline Monte Carlo equivalence metrics using dependence-preserving blocks.

Input axes: independent run, ordered step, walker (one for HMC), quantity.
Walkers remain together within time blocks. These metrics do not certify mixing.
"""
import numpy as np
from scipy.stats import t as student_t


def compare_positive_probabilities(candidate,reference,*,candidate_block_steps,reference_block_steps,family_size,look_index=1,diagnostics_validated=False,conditional_probabilities=False):
    """Sign-probability equivalence with time blocks that keep walkers together.

    A constant indicator cannot demonstrate rare-tail precision. Such a case
    stays INCONCLUSIVE rather than assigning zero Monte Carlo uncertainty.
    """
    a=np.asarray(candidate,dtype=float);b=np.asarray(reference,dtype=float)
    if a.ndim!=4 or b.ndim!=4 or a.shape[0]<4 or b.shape[0]<4 or a.shape[-1]!=b.shape[-1]:
        raise ValueError('Matching quantities and four independent runs required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite draws cannot be dropped')
    if conditional_probabilities and (np.any((a<0)|(a>1)) or np.any((b<0)|(b>1))):
        raise ValueError('Conditional probabilities must lie in [0,1]')
    if look_index<1 or family_size<a.shape[-1]:raise ValueError('Invalid look or underspecified multiplicity family')
    def estimate(x,steps):
        r,n,w,p=x.shape
        if n%steps or n//steps<5:raise ValueError('At least five complete equal time blocks per run; no discarded suffix')
        indicator=x if conditional_probabilities else (x>0).astype(float)
        blocks=indicator.reshape(r,n//steps,steps,w,p).mean((2,3)).reshape(-1,p)
        return indicator.mean((0,1,2)),blocks.var(0,ddof=1)/len(blocks),len(blocks)-1
    pa,va,dfa=estimate(a,candidate_block_steps);pb,vb,dfb=estimate(b,reference_block_steps)
    variance=va+vb;denominator=va**2/dfa+vb**2/dfb
    df=np.full_like(variance,min(dfa,dfb));np.divide(variance**2,denominator,out=df,where=denominator>0)
    alpha=.05/(look_index*(look_index+1))/family_size
    half=student_t.ppf(1-alpha/2,df)*np.sqrt(variance)
    difference=pa-pb;lo=difference-half;hi=difference+half
    # Even one unsampled tail invalidates the corresponding variance estimate.
    unresolved=(va==0)|(vb==0)
    lo=np.where(unresolved,-1.,lo);hi=np.where(unresolved,1.,hi)
    passed=(lo>-.01)&(hi<.01);outside=(hi<-.01)|(lo>.01)
    status='PASS' if passed.all() else ('FAIL' if outside.any() else 'INCONCLUSIVE')
    return dict(status=status if diagnostics_validated else 'INCONCLUSIVE',eligible_for_qualification=diagnostics_validated,
        metric_status=np.where(passed,'PASS',np.where(outside,'FAIL','INCONCLUSIVE')).tolist(),difference=difference.tolist(),
        interval_lower=lo.tolist(),interval_upper=hi.tolist(),mcse=np.where(unresolved,np.nan,np.sqrt(variance)).tolist(),
        unresolved_constant_indicators=unresolved.tolist(),per_interval_alpha=alpha,family_size=family_size,look_index=look_index,
        candidate_probability=pa.tolist(),reference_probability=pb.tolist(),margin=.01,
        estimator='bounded_probability_values' if conditional_probabilities else 'sampled_contrast_indicator')


def _statistics(x):
    flat=x.reshape(-1,x.shape[-1])
    return np.stack([flat.mean(0),*np.quantile(flat,[.05,.5,.95],axis=0),np.log(flat.std(0,ddof=1))])


def _block_variance(x,block_steps):
    r,n,w,p=x.shape
    if n%block_steps or n//block_steps<5:
        raise ValueError('At least five complete equal time blocks per run; no discarded suffix')
    blocks=x.reshape(r,n//block_steps,block_steps,w,p)
    stats=np.stack([_statistics(b) for run in blocks for b in run])
    return stats.var(0,ddof=1)/len(stats),len(stats)-1


def compare_scalar_draws(candidate,reference,*,candidate_block_steps,reference_block_steps,family_size,look_index=1,diagnostics_validated=False):
    """Bonferroni plus alpha spending, with conservative block-t intervals.

    Mean/quantile margins .05/.10 reference SD; SD ratio [.95,1.05].
    Each formal look spends .05/[k(k+1)] across the frozen comparison family.
    Exact equality/near-zero reference variance requires a separate prespecified
    absolute rule; this function will not substitute a numerical floor.
    """
    a=np.asarray(candidate,dtype=float);b=np.asarray(reference,dtype=float)
    if a.ndim!=4 or b.ndim!=4 or a.shape[0]<4 or b.shape[0]<4 or a.shape[-1]!=b.shape[-1]:
        raise ValueError('Matching quantities and four independent runs required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite draws cannot be dropped')
    if look_index<1 or family_size<5*a.shape[-1]:raise ValueError('Invalid look or underspecified multiplicity family')
    sd=b.reshape(-1,b.shape[-1]).std(0,ddof=1)
    if np.any(sd<=0) or np.any(a.reshape(-1,a.shape[-1]).std(0,ddof=1)<=0):
        raise ValueError('Deterministic or stuck quantity requires a separate validated rule')
    sa,sb=_statistics(a),_statistics(b);va,dfa=_block_variance(a,candidate_block_steps);vb,dfb=_block_variance(b,reference_block_steps)
    variance=va+vb
    denominator=va**2/dfa+vb**2/dfb
    df=np.full_like(variance,min(dfa,dfb));np.divide(variance**2,denominator,out=df,where=denominator>0)
    alpha=.05/(look_index*(look_index+1))/family_size
    half=student_t.ppf(1-alpha/2,df)*np.sqrt(variance)
    difference=sa-sb;lo=difference-half;hi=difference+half
    margins=np.stack([.05*sd,.1*sd,.1*sd,.1*sd])
    passed=np.concatenate([(lo[:4]>-margins)&(hi[:4]<margins),((lo[4]>np.log(.95))&(hi[4]<np.log(1.05)))[None]],axis=0)
    outside=np.concatenate([(hi[:4]<-margins)|(lo[:4]>margins),((hi[4]<np.log(.95))|(lo[4]>np.log(1.05)))[None]],axis=0)
    metric_status=np.where(passed,'PASS',np.where(outside,'FAIL','INCONCLUSIVE'))
    status='PASS' if passed.all() else ('FAIL' if outside.any() else 'INCONCLUSIVE')
    return dict(status=status if diagnostics_validated else 'INCONCLUSIVE',eligible_for_qualification=diagnostics_validated,metric_status=metric_status.tolist(),metric_pass=passed.tolist(),difference=difference.tolist(),interval_lower=lo.tolist(),interval_upper=hi.tolist(),mcse=np.sqrt(variance).tolist(),statistics=['mean','q05','q50','q95','log_sd_ratio'],family_size=family_size,look_index=look_index,per_interval_alpha=alpha,block_steps=[candidate_block_steps,reference_block_steps],reference_sd=sd.tolist())
