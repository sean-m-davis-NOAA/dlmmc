import numpy as np
import pytest
from dlmmc.tuning import walker_candidates, tuning_window


def test_counts_use_full_moved_dimension():
    assert walker_candidates(4) == [64, 128, 256]
    assert walker_candidates(64) == [128, 256, 512]
    assert walker_candidates(260) == [512, 1024, 2048]
    with pytest.raises(ValueError):
        walker_candidates(0)


def test_screen_preserves_dependence_and_rejects_false_agreement():
    rng = np.random.default_rng(841)
    x = rng.normal(size=(4, 1600, 8, 2))
    good = tuning_window(x)
    assert good['eligible']
    duplicate = tuning_window(np.repeat(x, 2, axis=2))
    np.testing.assert_allclose(good['minimum_ess'], duplicate['minimum_ess'], rtol=1e-4)
    separated = x.copy()
    separated[0, :, :, 0] += 2
    assert not tuning_window(separated)['eligible']
    expanding = x.copy()
    expanding[:, :800] *= .01
    assert not tuning_window(expanding)['eligible']
    assert not tuning_window(x[:, :16])['eligible']
    with pytest.raises(ValueError):
        tuning_window(x, target_ess=999)
