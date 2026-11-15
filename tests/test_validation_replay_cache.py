import hashlib
import json
import numpy as np
import pytest
from dlmmc.validation_cache import reuse_reconstruction_prefix
from test_validation_cache import fixture


def replay_fixture(root, change=None):
    previous, destination, new, new_config = fixture(root)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    old = json.loads((root/'old-cert.json').read_text())
    old.update(chunk_sha256={}, total_walkers=2, warmup_transitions=0)
    new.update(sampling_directory='replayed', cell_index=0, chunk_sha256={}, total_walkers=2, warmup_transitions=0)
    for run in range(4):
        x = np.arange(48, dtype=float).reshape(8,2,3)+run
        for name, data in [('sampling',x), ('replayed',x[:,:,None,:])]:
            path = root/name/f'run-{run}'/'chunk-000000-000008.npz'
            path.parent.mkdir(parents=True,exist_ok=True)
            warmup = np.zeros(8,bool)
            if name == 'replayed' and run == 1:
                if change == 'draw': data=data.copy(); data[0,0,0,0]+=1
                if change == 'warmup': warmup[0]=True
            np.savez(path,draws=data,warmup=warmup,logp=np.zeros(data.shape[:2]))
            (old if name == 'sampling' else new)['chunk_sha256'][str(path.relative_to(root))] = sha(path)
    (root/'old-cert.json').write_text(json.dumps(old))
    (root/'new-cert.json').write_text(json.dumps(new))
    config = json.loads((previous/'config.json').read_text())
    config['source_hashes']['old-cert.json'] = sha(root/'old-cert.json')
    (previous/'config.json').write_text(json.dumps(config))
    done = json.loads((previous/'complete.json').read_text())
    done['config_sha256'] = sha(previous/'config.json')
    (previous/'complete.json').write_text(json.dumps(done))
    new_config['source_hashes']['new-cert.json'] = sha(root/'new-cert.json')
    replay = dict(replay_source='sampling', replay_manifest=old['chunk_sha256'])
    if change == 'source': replay['replay_source']='unrelated'
    (root/'replayed/config.json').write_text(json.dumps(replay))
    receipt = dict(status='PASS',config_sha256=sha(root/'replayed/config.json'),steps_per_run=8,runs=4,warmup=0)
    if change == 'short': receipt['steps_per_run']=4
    if change == 'failed': receipt['status']='FAIL'
    (root/'replay.json').write_text(json.dumps(receipt))
    return previous,destination,new,new_config


def test_replayed_draws_reuse_identical_conditional_outputs(tmp_path):
    args = replay_fixture(tmp_path)
    receipt = reuse_reconstruction_prefix(tmp_path,*args,replay_receipt='replay.json')
    assert len(receipt['exact_replay_proof']['verified_parameter_chunks']) == 4
    for path in args[0].glob('*.npz'):
        assert path.stat().st_ino == (args[1]/path.name).stat().st_ino
    assert reuse_reconstruction_prefix(tmp_path,*args,replay_receipt='replay.json') == receipt


@pytest.mark.parametrize('change',['draw','warmup','source','short','failed'])
def test_replay_cannot_license_different_conditioning_draws(tmp_path,change):
    args = replay_fixture(tmp_path,change)
    with pytest.raises(ValueError):
        reuse_reconstruction_prefix(tmp_path,*args,replay_receipt='replay.json')
    assert not (args[1]/'prefix-reuse.json').exists()


def test_cross_cache_requires_explicit_replay_receipt(tmp_path):
    args = replay_fixture(tmp_path)
    with pytest.raises(ValueError,match='sampling_directory'):
        reuse_reconstruction_prefix(tmp_path,*args)
