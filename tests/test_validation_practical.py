import numpy as np
import pytest
from dlmmc.validation_practical import practical_consistency


def assess(d, lo, hi, **kwargs):
    return practical_consistency(d, lo, hi, -.05, .05,
                                 diagnostics_validated=True, **kwargs)


def test_compatibility_is_not_strict_equivalence():
    r = assess(.02, -.06, .10, ess_validated=True)
    assert r['status'] == 'PRACTICALLY_CONSISTENT_AT_ACHIEVED_MC_PRECISION'
    assert not r['strict_interval_containment']


def test_small_resolved_bias_is_scientifically_compatible():
    assert assess(.01, .005, .015, ess_validated=True)['eligible']
    assert assess(.01, .005, .015, ess_validated=True)['metric_status'] == 'MC_COMPATIBLE'


def test_resolved_large_difference_requires_investigation():
    r = assess(.12, .08, .16, ess_validated=True)
    assert r['status'] == 'INVESTIGATE'
    assert r['point_estimate_outside_margin']


def test_uncertain_point_outside_margin_remains_flagged():
    r = assess(.06, .02, .10, ess_validated=True)
    assert r['metric_status'] == 'MC_COMPATIBLE'
    assert r['point_estimate_outside_margin']


def test_no_ess_no_practical_pass_even_with_huge_interval():
    assert assess(.01, -100, 100)['status'] == 'INCONCLUSIVE'


def test_missing_mixing_cannot_pass():
    r = practical_consistency(0, -.01, .01, -.05, .05, ess_validated=True)
    assert r['status'] == 'INCONCLUSIVE'


@pytest.mark.parametrize('args', [(np.nan, 0, 1), (0, 1, 0), (0, -np.inf, 1)])
def test_invalid_values_are_not_silently_discarded(args):
    with pytest.raises(ValueError):
        assess(*args, ess_validated=True)


def test_one_bad_quantity_is_not_averaged_away():
    r = assess([0, .12], [-.01, .08], [.01, .16], ess_validated=True)
    assert r['status'] == 'INVESTIGATE'
    assert r['metric_status'] == ['MC_COMPATIBLE', 'INVESTIGATE']
