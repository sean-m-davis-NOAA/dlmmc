from dataclasses import replace
import json
import numpy as np
import pytest

from dlm_grid_benchmark import load_basic_grid, sampling_timings, PROFILE, ROOT
from dlmmc.examples import load_panels


def test_full_basic_selection_matches_historical_grid_and_tutorial(monkeypatch):
    pytest.importorskip("netCDF4")
    monkeypatch.chdir(ROOT)
    y, s, x, metadata = load_basic_grid()
    historical = json.loads(
        (ROOT / "examples/modernization/current/16-gpu-comparison-results.json").read_text()
    )
    assert y.shape == s.shape == (396, 312)
    assert x.shape == (312, 396, 5)
    assert metadata["panel_ids"] == [row["panel"] for row in historical["panels"]]
    assert len(metadata["excluded_zero_placeholders"]) == 12
    fy, fs, fx, _ = load_panels(ROOT / "examples/modernization")
    np.testing.assert_array_equal(y.time, fy.time)
    for panel in fy.panel.values:
        for full, frozen in ((y, fy), (s, fs), (x, fx)):
            np.testing.assert_allclose(full.sel(panel=panel), frozen.sel(panel=panel),
                                       rtol=1e-14, atol=1e-16)
        np.testing.assert_array_equal(y.sel(panel=panel).pressure, fy.sel(panel=panel).pressure)
        np.testing.assert_array_equal(y.sel(panel=panel).latitude, fy.sel(panel=panel).latitude)


@pytest.fixture
def checkpoints(tmp_path):
    profile = replace(PROFILE, half_walkers=4, warmup=2, retained_sweeps=4, chunk_sweeps=2)
    (tmp_path / "run-0").mkdir()
    (tmp_path / "latest.json").write_text(json.dumps({
        "status": "SAMPLING_COMPLETE_NOT_CONVERGENCE", "iterations": [6], "target_iteration": 6,
    }))
    for start, seconds in ((0, 1.5), (2, 2.5), (4, 3.5)):
        steps = np.arange(max(start, 2) + 1, start + 3)
        np.savez(tmp_path / "run-0" / f"block-{start:09d}-{start+2:09d}.npz",
                 metadata=json.dumps({"iteration": start + 2, "warmup": 2}),
                 retained_iterations=steps, draws=np.zeros((len(steps), 8, 3, 4)),
                 draw_logp=np.zeros((len(steps), 8, 3)), sampling_seconds=seconds)
    return tmp_path, profile


def test_sampling_times_include_burnin_and_all_retained_blocks(checkpoints):
    folder, profile = checkpoints
    assert sampling_timings(folder, profile, 3, 4) == {
        "burnin_seconds": 1.5, "retained_sampling_seconds": 6.0,
        "sampling_including_burnin_seconds": 7.5,
    }
    with pytest.raises(ValueError, match="Invalid retained checkpoint"):
        sampling_timings(folder, profile, 4, 4)


@pytest.mark.parametrize("failure", ["wall_cap", "missing_middle", "missing_end"])
def test_incomplete_runs_cannot_report_success(checkpoints, failure):
    folder, profile = checkpoints
    if failure == "wall_cap":
        (folder / "latest.json").write_text(json.dumps({
            "status": "WALL_CAP_CHECKPOINTED", "iterations": [4], "target_iteration": 6,
        }))
    else:
        start = 2 if failure == "missing_middle" else 4
        (folder / "run-0" / f"block-{start:09d}-{start+2:09d}.npz").unlink()
    with pytest.raises((ValueError, RuntimeError), match="Incomplete|Noncontiguous"):
        sampling_timings(folder, profile, 3, 4)
