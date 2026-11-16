import numpy as np
import pytest

from dlm_cmdstan_benchmark import extrapolate, select_cells


def test_subset_is_reproducible_and_covers_every_stratum():
    indices = select_cells(312, 16, 2026111601)
    np.testing.assert_array_equal(indices, select_cells(312, 16, 2026111601))
    for index, group in zip(indices, np.array_split(np.arange(312), 16)):
        assert index in group
    assert len(set(indices)) == 16
    with pytest.raises(ValueError, match="proper subset"):
        select_cells(312, 312, 1)


def test_extrapolation_counts_all_chains_and_charges_setup_only_once():
    result = extrapolate(50, [10] * 8, [8] * 8, cells=4, workers=2,
                         setup_seconds=7, summary_seconds=2)
    assert result["batch_scaled_seconds"] == 3900
    assert result["batch_scaled_plus_one_setup_seconds"] == 3907
    assert result["pipeline_scaled_plus_one_setup_seconds"] == 4063
    assert result["ideal_load_balance_seconds"] == 3120
    assert result["stan_reported_time_at_ideal_load_balance_seconds"] == 2496
    assert result["busy_worker_fraction"] == 0.8
    assert result["kind"] == "EXTRAPOLATED_NOT_MEASURED_FULL_GRID"


@pytest.mark.parametrize("task_times,stan_times", [
    ([], []), ([1, 2, 3], [1, 2, 3]), ([1, 2], [1]), ([np.nan, 2], [1, 2]),
    ([0, 2], [1, 2]),
])
def test_partial_or_invalid_measurements_are_not_extrapolated(task_times, stan_times):
    with pytest.raises(ValueError, match="Complete positive"):
        extrapolate(5, task_times, stan_times, cells=2, workers=2, setup_seconds=1)
