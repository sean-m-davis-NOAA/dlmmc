import numpy as np
import pytest
from dlmmc.validation_selection import stratified_indices, selected_diagnostics


def test_indices_stratify_without_replacement_and_are_reproducible():
    a = stratified_indices(8000, 128, 2048, 100)
    np.testing.assert_array_equal(a, stratified_indices(8000, 128, 2048, 100))
    assert np.all(np.diff(a[:, 0]) > 0)
    assert a[:, 1].min() >= 0 and a[:, 1].max() < 128
    assert a[0, 0] < 4 and a[-1, 0] >= 7996


def test_oversampling_retained_times_rejected():
    with pytest.raises(ValueError):
        stratified_indices(10, 128, 11, 4)


def test_iid_draws_have_sufficient_effective_precision():
    d = selected_diagnostics(np.random.default_rng(42).normal(size=(4, 2048, 3)))
    assert d['eligible'] and d['minimum_ess'] > 2000


def test_separated_runs_are_not_certified_by_raw_count():
    x = np.random.default_rng(42).normal(size=(4, 2048, 3))
    x[0] += 4
    assert not selected_diagnostics(x)['eligible']


def test_slow_dependence_is_not_independent_samples():
    x = np.cumsum(np.random.default_rng(42).normal(size=(4, 2048, 3)), axis=1)
    assert not selected_diagnostics(x)['eligible']
