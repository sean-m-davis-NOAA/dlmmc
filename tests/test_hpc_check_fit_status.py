import sys

import numpy as np
import pytest
import xarray as xr

from experiments.hpc import check_fit_status


def _run_check(monkeypatch, status, indices):
    result = xr.Dataset(
        {'fit_status': (('pressure', 'latitude'), np.asarray(status))},
    )
    monkeypatch.setattr(check_fit_status.xr, 'open_dataset', lambda path: result)
    monkeypatch.setattr(
        sys, 'argv', ['check_fit_status.py', 'result.nc', *map(str, indices)]
    )
    check_fit_status.main()


def test_check_fit_status_accepts_exact_selected_cells(monkeypatch, capsys):
    _run_check(monkeypatch, [[1, 0, 0], [0, 1, 0]], [0, 4])
    assert 'fit_status_counts' in capsys.readouterr().out


@pytest.mark.parametrize(
    'status',
    [
        [[1, 0, 0], [0, -1, 0]],
        [[1, -1, 0], [0, 1, 0]],
        [[1, 1, 0], [0, 1, 0]],
    ],
)
def test_check_fit_status_rejects_failed_or_unselected_cells(monkeypatch, status):
    with pytest.raises(SystemExit, match='Unexpected fit_status'):
        _run_check(monkeypatch, status, [0, 4])


@pytest.mark.parametrize('indices', [[0, 0], [-1], [6]])
def test_check_fit_status_rejects_invalid_indices(monkeypatch, indices):
    with pytest.raises(SystemExit) as exc:
        _run_check(monkeypatch, [[0, 0, 0], [0, 0, 0]], indices)
    assert exc.value.code == 2
