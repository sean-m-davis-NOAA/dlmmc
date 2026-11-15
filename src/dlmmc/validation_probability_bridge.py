"""Explicit numerical compatibility proof for one offline CDF cache transition.

This is not bitwise engine identity. It permits only a qualified cached-Torch
CDF prefix followed by qualified Numba CDF evaluations of the same estimator.
State samples, changed parameters, arbitrary source edits and changed policies
remain ineligible. The ordinary strict cache path is unchanged.
"""
import hashlib
import json
from pathlib import Path

CACHED = 'checkouts/dlmmc/src/dlmmc/conditional_statistics_cached.py'
NUMBA = 'checkouts/dlmmc/src/dlmmc/numba_conditional.py'
QUALIFICATIONS = tuple('reports/'+name for name in (
    '01-numba-conditional-real-panels.json',
    '01-numba-conditional-prefix-profile.json',
    '01-numba-conditional-rare-sign.json'))
SPEC = 'configs/stage01-cdf-prefix-bridge.json'


def validate_probability_bridge(root, previous, old_config, new_config, old_certificate,
                                new_certificate, old_hashes, new_hashes, bridge_path):
    root, previous = Path(root), Path(previous)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    bridge_path = root/bridge_path
    bridge = json.loads(bridge_path.read_text())
    expected = dict(path=str(bridge_path.relative_to(root)), sha256=sha(bridge_path))
    if new_config.get('probability_prefix_bridge') != expected:
        raise ValueError('Explicit bridge provenance missing or changed')
    if old_config.get('seed') is not None or new_config.get('seed') is not None:
        raise ValueError('Probability bridge cannot reuse stochastic state samples')
    for key in ('selected_walkers', 'estimator_revision', 'method', 'cdf', 'plot_selection', 'steps_per_chunk'):
        if key not in old_config or old_config[key] != new_config.get(key):
            raise ValueError('Probability estimator changed: '+key)
    if old_config.get('calculation_engine') is not None or new_config.get('calculation_engine') != 'numba-full-joseph-float64':
        raise ValueError('Unsupported probability engine transition')
    if set(old_hashes)-set(new_hashes) != {CACHED} or set(new_hashes)-set(old_hashes) != {NUMBA, *QUALIFICATIONS}:
        raise ValueError('Unexpected probability source transition')
    for key in set(old_hashes) & set(new_hashes):
        if old_hashes[key] != new_hashes[key]:
            raise ValueError('Common probability equations/data changed')
    for sources in (old_hashes, new_hashes):
        for path, value in sources.items():
            if sha(root/path) != value:
                raise ValueError('Probability source evidence changed: '+path)
    for path in QUALIFICATIONS:
        evidence = json.loads((root/path).read_text())
        hashes = evidence['source_sha256']
        digest = hashes if isinstance(hashes, str) else hashes[NUMBA]
        if evidence['status'] != 'PASS' or digest != new_hashes[NUMBA]:
            raise ValueError('Numba CDF numerical qualification missing')
    if (bridge.get('status') != 'PASS'
            or bridge.get('previous') != str(previous.relative_to(root))
            or bridge.get('old_config_sha256') != sha(previous/'config.json')
            or bridge.get('new_certificate_sha256') != sha(root/new_config['certificate'])
            or bridge.get('numba_source_sha256') != new_hashes[NUMBA]
            or bridge.get('spec_sha256') != sha(root/SPEC)):
        raise ValueError('Bridge does not describe these exact windows/engines')
    spec = json.loads((root/SPEC).read_text())
    for sources in (bridge['implementation_sha256'], bridge['qualification_sha256']):
        for path, digest in sources.items():
            if sha(root/path) != digest:
                raise ValueError('Bridge validation implementation/evidence changed')
    count = old_certificate['retained_steps']
    size = old_config['steps_per_chunk']
    expected_locations = {(r, (i*(count//size-1)//(spec['chunks_per_run']-1))*size)
                          for r in range(old_certificate['independent_runs'])
                          for i in range(spec['chunks_per_run'])}
    rows = bridge['checks']
    if len(rows) != len(expected_locations) or {(r['run'], r['start']) for r in rows} != expected_locations:
        raise ValueError('Incomplete prespecified CDF bridge checks')
    for row in rows:
        if row['status'] != 'PASS' or not 0 <= row['maximum_tolerance_ratio'] <= 1:
            raise ValueError('CDF prefix numerical comparison failed')
        path = previous/f'run-{row["run"]}-chunk-{row["start"]:06d}.npz'
        if sha(path) != row['cached_chunk_sha256']:
            raise ValueError('Checked CDF prefix chunk changed')
    return dict(**expected, scope='Numerical CDF compatibility, not bitwise engine identity',
                previous_engine='cached-Torch-float64', new_engine='numba-full-joseph-float64',
                checked_chunks=len(rows), spec_sha256=bridge['spec_sha256'])
