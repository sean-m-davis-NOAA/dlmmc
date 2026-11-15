import itertools

import numpy as np
import pytest

from dlmmc.validation_calibration import (
    coverage_smoke_test, exact_uniform_rank_test,
    posterior_coverage_membership, randomized_rank,
)


def test_exact_multinomial_against_brute_force_sequences():
    # Independent enumeration of all3**4 equally likely sequences.
    sequences = list(itertools.product(range(3), repeat=4))
    counts = [tuple(np.bincount(s, minlength=3)) for s in sequences]
    frequency = {c: counts.count(c) for c in set(counts)}
    for observed, multiplicity in frequency.items():
        expected = sum(v for v in frequency.values() if v <= multiplicity) / len(sequences)
        assert exact_uniform_rank_test(observed) == expected


def test_twenty_dataset_rank_extremes():
    assert exact_uniform_rank_test([4]*5) == 1
    assert exact_uniform_rank_test([20,0,0,0,0]) == 5/5**20


@pytest.mark.parametrize('counts', [[1.5,2], [-1,2], [21,0], [0,0], [np.nan,2]])
def test_rank_invalid_counts(counts):
    with pytest.raises(ValueError): exact_uniform_rank_test(counts)


def test_randomized_ties_reproducible_and_not_walkers():
    rng = np.random.default_rng(14)
    values = [randomized_rank(1, [0,1,1,2], rng)[0] for _ in range(100)]
    assert set(values) == {1,2,3}
    assert randomized_rank(1, [0,1,1,2], np.random.default_rng(4)) == randomized_rank(1, [0,1,1,2], np.random.default_rng(4))
    with pytest.raises(ValueError): randomized_rank(0, np.zeros(128), rng)


def test_dependence_preserving_endpoint_bands_and_ambiguity():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(4,1000,1))
    duplicated_walkers = np.repeat(x, 7, axis=2)
    a = posterior_coverage_membership(x, 0, block_steps=100, alpha=.001)
    b = posterior_coverage_membership(duplicated_walkers, 0, block_steps=100, alpha=.001)
    assert a['membership'] == b['membership'] == 'COVERED'
    # Duplicating walkers cannot create an artificial sqrt(7) precision gain.
    assert np.allclose(a['endpoint_mcse'], b['endpoint_mcse'], rtol=.15)
    edge = a['posterior_interval'][0]
    assert posterior_coverage_membership(x, edge, block_steps=100, alpha=.001)['membership'] == 'INCONCLUSIVE'
    assert posterior_coverage_membership(x, 10, block_steps=100, alpha=.001)['membership'] == 'NOT_COVERED'
    with pytest.raises(ValueError): posterior_coverage_membership(x, 0, block_steps=99, alpha=.001)
    assert posterior_coverage_membership(np.zeros_like(x), 0, block_steps=100, alpha=.001)['membership'] == 'INCONCLUSIVE'


def test_unknown_datasets_cannot_manufacture_pass():
    assert coverage_smoke_test(['COVERED']*19+['NOT_COVERED'],alpha=.05/52)['status'] == 'PASS'
    assert coverage_smoke_test(['COVERED']*19+['INCONCLUSIVE'],alpha=.05/52)['status'] == 'INCONCLUSIVE'
    assert coverage_smoke_test(['NOT_COVERED']*19+['INCONCLUSIVE'],alpha=.05/52)['status'] == 'FAIL'
    with pytest.raises(ValueError): coverage_smoke_test(['COVERED']*19,alpha=.05/52)
