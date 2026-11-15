import json
import pytest
from dlmmc.validation_timing import timing_exclusions


def test_pauses_exclude_only_named_timings_and_union_history(tmp_path):
    reports = tmp_path / 'reports'
    reports.mkdir()
    assert timing_exclusions(tmp_path) == (set(), {})
    for suffix, entries in [('', ['run/a.npz']), ('-second', ['run/b.npz'])]:
        (reports / f'resumed-processes{suffix}.json').write_text(json.dumps(dict(
            status='RESUMED', timing_exclusions=entries,
            paused_cmdstan_timing_exclusion='stan/chain.csv')))
    paths, provenance = timing_exclusions(tmp_path)
    assert paths == {'run/a.npz', 'run/b.npz', 'stan/chain.csv'}
    assert len(provenance) == 2
    assert all(len(value) == 64 for value in provenance.values())


@pytest.mark.parametrize('path', ['/absolute/file', '../other/file'])
def test_reject_ambiguous_pause_paths(tmp_path, path):
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'reports/resumed-processes.json').write_text(json.dumps(dict(
        status='RESUMED', timing_exclusions=[path])))
    with pytest.raises(ValueError, match='project-relative'):
        timing_exclusions(tmp_path)


def test_observed_system_sleep_does_not_require_a_user_resume(tmp_path):
    reports = tmp_path/'reports'; reports.mkdir()
    (reports/'timing-interruptions-sleep.json').write_text(json.dumps(dict(
        status='OBSERVED_SYSTEM_SLEEP', timing_exclusions=['run/sleep.npz', 'stan/chain.csv'])))
    paths, provenance = timing_exclusions(tmp_path)
    assert paths == {'run/sleep.npz', 'stan/chain.csv'}
    assert set(provenance) == {'reports/timing-interruptions-sleep.json'}
