import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("arviz")

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "hpc" / "cell_diagnostics.py"
spec = importlib.util.spec_from_file_location("cell_diagnostics", SCRIPT)
cell_diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cell_diagnostics)


def _dataset(nchains=4):
    rng = np.random.default_rng(0)
    shape = (nchains, 200, 2, 3)
    status = np.zeros((2, 3), dtype=np.int8)
    status[0, 1] = status[1, 2] = 1
    sigma = rng.normal(size=shape)
    sigma[0, :, 1, 2] += 10.0  # one stuck chain in cell (1, 2)
    return xr.Dataset(
        {
            "sigma": (("chain", "draw", "pressure", "latitude"), sigma),
            "beta": (("chain", "draw", "regressor", "pressure", "latitude"), rng.normal(size=(nchains, 200, 2, 2, 3))),
            "trend": (("chain", "draw", "time", "pressure", "latitude"), rng.normal(size=(nchains, 200, 4, 2, 3))),
            "fit_status": (("pressure", "latitude"), status),
        }
    )


def test_rows_cover_only_fitted_cells_and_flag_bad_mixing():
    rows = cell_diagnostics.cell_diagnostics(_dataset(), "run")

    assert [r["flat_cell_id"] for r in rows] == [1, 5]
    assert float(rows[0]["max_rhat"]) < 1.05
    assert float(rows[1]["max_rhat"]) > 1.5
    assert "trend" not in rows[0]["diagnostic_source"]
    assert "beta" in rows[0]["diagnostic_source"]


def test_single_chain_leaves_rhat_blank():
    rows = cell_diagnostics.cell_diagnostics(_dataset(nchains=1), "run")

    assert all(r["max_rhat"] == "" for r in rows)


def test_divergences_are_read_when_stored():
    ds = _dataset()
    assert all(r["divergences"] == "" for r in cell_diagnostics.cell_diagnostics(ds, "run"))

    counts = np.full((2, 3), np.nan)
    counts[0, 1], counts[1, 2] = 0, 7
    ds["divergent_transitions"] = (("pressure", "latitude"), counts)
    rows = cell_diagnostics.cell_diagnostics(ds, "run")

    assert [r["divergences"] for r in rows] == [0, 7]
    assert "divergent_transitions" not in rows[0]["diagnostic_source"]
    assert rows[0]["notes"] == ""
