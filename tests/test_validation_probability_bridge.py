import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from dlmmc.validation_cache import reuse_reconstruction_prefix
from dlmmc.validation_probability_bridge import CACHED, NUMBA, QUALIFICATIONS, SPEC
from test_validation_cache import fixture


def bridge_fixture(root):
    previous, destination, new, new_config = fixture(root)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    def write(path, value):
        path = root/path; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); return sha(path)
    old = json.loads((root/'old-cert.json').read_text())
    old['retained_steps'] = 16; new['retained_steps'] = 24
    old_hash = write('old-cert.json', old); write('new-cert.json', new)
    old_config = json.loads((previous/'config.json').read_text())
    policy = dict(seed=None, selected_walkers=[0, 1], estimator_revision=2,
                  method='conditional Gaussian CDF', cdf='stable float64', plot_selection='none')
    old_config.update(policy); new_config.update(policy)
    old_config['source_hashes']['old-cert.json'] = old_hash
    new_config['source_hashes']['new-cert.json'] = sha(root/'new-cert.json')
    old_config['source_hashes'][CACHED] = write(CACHED, 'cached implementation')
    new_config['source_hashes'][NUMBA] = write(NUMBA, 'numba implementation')
    for path in QUALIFICATIONS:
        new_config['source_hashes'][path] = write(path, dict(status='PASS', source_sha256={NUMBA:sha(root/NUMBA)}))
    new_config['calculation_engine'] = 'numba-full-joseph-float64'
    write('old/config.json', old_config)
    write('old/complete.json', dict(config_sha256=sha(previous/'config.json'), retained_steps=16, runs=4))
    for run in range(4):
        for start in (0, 8):
            np.savez(previous/f'run-{run}-chunk-{start:06d}.npz', probabilities=np.full((8, 2, 3), .5))
    write(SPEC, dict(chunks_per_run=2))
    bridge = dict(status='PASS', previous='old', old_config_sha256=sha(previous/'config.json'),
                  new_certificate_sha256=sha(root/'new-cert.json'), numba_source_sha256=sha(root/NUMBA),
                  spec_sha256=sha(root/SPEC), implementation_sha256={}, qualification_sha256={},
                  checks=[dict(run=r, start=s, status='PASS', maximum_tolerance_ratio=0.,
                               cached_chunk_sha256=sha(previous/f'run-{r}-chunk-{s:06d}.npz'))
                          for r in range(4) for s in (0, 8)])
    write('bridge.json', bridge)
    new_config['probability_prefix_bridge'] = dict(path='bridge.json', sha256=sha(root/'bridge.json'))
    return previous, destination, new, new_config


def test_explicit_bridge_keeps_original_probability_bytes(tmp_path):
    args = bridge_fixture(tmp_path)
    # Ordinary cache semantics still reject differing engine provenance.
    with pytest.raises(ValueError, match='equations/data'):
        reuse_reconstruction_prefix(tmp_path, *args)
    receipt = reuse_reconstruction_prefix(tmp_path, *args, probability_bridge='bridge.json')
    assert receipt['status'] == 'VERIFIED_NUMERICALLY_COMPATIBLE_PROBABILITY_PREFIX'
    assert receipt['probability_bridge']['checked_chunks'] == 8
    for path in args[0].glob('*.npz'):
        assert path.stat().st_ino == (args[1]/path.name).stat().st_ino


@pytest.mark.parametrize('change', ['seed', 'method', 'data', 'qualification', 'probability', 'parameters', 'missing_check', 'failed_check', 'bridge_hash'])
def test_bridge_cannot_hide_changed_or_unqualified_evidence(tmp_path, change):
    previous, destination, new, config = bridge_fixture(tmp_path)
    if change == 'seed': config['seed'] = 17
    if change == 'method': config['method'] = 'different estimator'
    if change == 'data': (tmp_path/'data.json').write_text('changed')
    if change == 'qualification': (tmp_path/QUALIFICATIONS[0]).write_text('{}')
    if change == 'probability': np.savez(previous/'run-0-chunk-000000.npz', probabilities=np.zeros(8))
    if change == 'parameters': np.savez(tmp_path/'sample.npz', draws=np.arange(9))
    if change in ('missing_check', 'failed_check'):
        path = tmp_path/'bridge.json'; report = json.loads(path.read_text())
        if change == 'missing_check': report['checks'].pop()
        else: report['checks'][0]['status'] = 'FAIL'
        path.write_text(json.dumps(report))
        config['probability_prefix_bridge']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if change == 'bridge_hash': config['probability_prefix_bridge']['sha256'] = 'wrong'
    with pytest.raises(ValueError):
        reuse_reconstruction_prefix(tmp_path, previous, destination, new, config, probability_bridge='bridge.json')
    assert not (destination/'prefix-reuse.json').exists()
