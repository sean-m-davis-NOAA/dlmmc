"""Narrow, reviewed coverage refinement; never recompute frozen SBC ranks."""
from copy import deepcopy

from .validation_calibration import coverage_smoke_test

WINDOWS = (32000, 64000, 128000, 256000)


def endpoint_alpha(retained):
    if retained not in WINDOWS:
        raise ValueError('Outside approved coverage refinement windows')
    look = WINDOWS.index(retained) + 3  # k1=8000 and k2=16000 already spent.
    return .05 / (look * (look + 1)) / 1040


def merge_coverage(baseline, coverage, *, retained):
    """Replace only dataset18 sigma_seas coverage, preserving every rank."""
    if (baseline['model'] != 'vanilla_ar1' or baseline['retained_steps'] != 16000
            or [d['replication'] for d in baseline['datasets']] != list(range(20))
            or any(d['status'] != 'ANALYZED_NOT_MODEL_QUALIFICATION' for d in baseline['datasets'])):
        raise ValueError('Complete frozen vanilla16k baseline required')
    if coverage['endpoint_alpha'] != endpoint_alpha(retained):
        raise ValueError('Endpoint alpha must continue the original spending sequence')
    result = deepcopy(baseline)
    quantities = result['datasets'][18]['quantities']
    j = next(i for i, q in enumerate(quantities) if q['name'] == 'sigma_seas')
    if quantities[j]['coverage']['membership'] != 'INCONCLUSIVE':
        raise ValueError('Only the originally ambiguous endpoint is authorized')
    quantities[j]['coverage'] = deepcopy(coverage)
    quantities[j]['coverage_retained_steps'] = retained
    aggregate = next(q for q in result['aggregate'] if q['name'] == 'sigma_seas')
    aggregate['coverage'] = coverage_smoke_test(
        [d['quantities'][j]['coverage']['membership'] for d in result['datasets']],
        alpha=.05/52)
    result['status'] = ('FAIL' if any(q['rank_status'] == 'FAIL' or q['coverage']['status'] == 'FAIL'
                                    for q in result['aggregate']) else
                        'INCONCLUSIVE' if any(q['coverage']['status'] == 'INCONCLUSIVE'
                                             for q in result['aggregate']) else 'PASS')
    result['retained_steps'] = 'mixed; see rank_retained_steps and coverage_refinement'
    result['rank_retained_steps'] = 16000
    result['coverage_refinement'] = dict(model='vanilla_ar1', replication=18,
                                       quantity='sigma_seas', retained_steps=retained)
    return result
