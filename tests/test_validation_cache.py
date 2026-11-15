import hashlib
import json
import numpy as np
import pytest
from dlmmc.validation_cache import reuse_reconstruction_prefix,reuse_reference_chains


def fixture(root):
    previous=root/'old';previous.mkdir();destination=root/'new'
    data=root/'data.json';data.write_text('{"data":1}')
    sample=root/'sample.npz';np.savez(sample,draws=np.arange(8))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    old=dict(status='WINDOW_DIAGNOSTICS_PASS',model='vanilla_ar1',panel='p05',sampling_directory='sampling',data_sha256=sha(data),warmup_transitions=100,independent_runs=4,total_walkers=128,retained_steps=8,chunk_sha256={'sample.npz':sha(sample)})
    (root/'old-cert.json').write_text(json.dumps(old));new=dict(old,retained_steps=16)
    (root/'new-cert.json').write_text(json.dumps(new))
    config=dict(certificate='old-cert.json',source_hashes={'data.json':sha(data),'old-cert.json':sha(root/'old-cert.json')},seed=17,steps_per_chunk=8,plot_selection='fixed')
    (previous/'config.json').write_text(json.dumps(config))
    (previous/'complete.json').write_text(json.dumps(dict(config_sha256=sha(previous/'config.json'),retained_steps=8,runs=4)))
    for run in range(4):np.savez(previous/f'run-{run}-chunk-000000.npz',summaries=np.arange(16).reshape(8,2)+run)
    (previous/'quantities.json').write_text('["trend"]')
    new_config=dict(config,certificate='new-cert.json',source_hashes={'data.json':sha(data),'new-cert.json':sha(root/'new-cert.json')})
    return previous,destination,new,new_config


def test_reuse_is_exact_and_idempotent(tmp_path):
    args=fixture(tmp_path);receipt=reuse_reconstruction_prefix(tmp_path,*args)
    assert receipt['retained_steps_reused']==8
    for src in args[0].glob('*.npz'):
        dst=args[1]/src.name
        assert dst.read_bytes()==src.read_bytes()
        assert dst.stat().st_ino==src.stat().st_ino
    assert reuse_reconstruction_prefix(tmp_path,*args)==receipt


@pytest.mark.parametrize('change',['data','sample','seed','missing_run','destination'])
def test_cannot_reuse_changed_evidence(tmp_path,change):
    previous,destination,new,config=fixture(tmp_path)
    if change=='data':(tmp_path/'data.json').write_text('{"data":2}')
    if change=='sample':np.savez(tmp_path/'sample.npz',draws=np.arange(8)+1)
    if change=='seed':config['seed']+=1
    if change=='missing_run':(previous/'run-3-chunk-000000.npz').unlink()
    if change=='destination':
        destination.mkdir();np.savez(destination/'run-0-chunk-000000.npz',summaries=np.zeros(8))
    with pytest.raises(ValueError):reuse_reconstruction_prefix(tmp_path,previous,destination,new,config)
    assert not (destination/'prefix-reuse.json').exists()


def reference_fixture(root,probabilities=False):
    old=root/'old';old.mkdir();new=root/'new'
    old_source=root/'old_source';old_source.mkdir();new_source=root/'new_source';new_source.mkdir()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for i in range(8):(new_source/f'chain-{i}.csv').write_text(f'parameter\n{i}\n')
    for i in range(4):(old_source/f'chain-{i}.csv').write_bytes((new_source/f'chain-{i}.csv').read_bytes())
    for folder in (old_source,new_source):(folder/'data.json').write_text('{"data":1}')
    config=dict(source_hashes={str(p.relative_to(root)):sha(p) for p in old_source.iterdir()},chunk_size=8,seed=17)
    new_config=dict(config,source_hashes={str(p.relative_to(root)):sha(p) for p in new_source.iterdir()})
    (old/'config.json').write_text(json.dumps(config))
    done=dict(config_sha256=sha(old/'config.json'),shape=[4,8,1,3]) if probabilities else dict(config_sha256=sha(old/'config.json'),chains=4,draws_per_chain=[8]*4)
    (old/'complete.json').write_text(json.dumps(done))
    for i in range(4):np.savez(old/f'run-{i}-chunk-000000.npz',values=np.arange(8)+i)
    (old/'quantities.json').write_text('["trend"]')
    return old,new,new_config


@pytest.mark.parametrize('probabilities',[False,True])
def test_reference_chain_reuse_preserves_all_old_draws(tmp_path,probabilities):
    args=reference_fixture(tmp_path,probabilities)
    result=reuse_reference_chains(tmp_path,*args,probabilities=probabilities)
    assert result['chains_reused']==4
    for f in args[0].glob('*.npz'):assert f.read_bytes()==(args[1]/f.name).read_bytes()


@pytest.mark.parametrize('change',['seed','order','data','missing_chunk'])
def test_reference_reuse_cannot_hide_changed_sampling_or_smoothing(tmp_path,change):
    old,new,config=reference_fixture(tmp_path)
    if change=='seed':config['seed']+=1
    if change=='order':
        config['source_hashes']['new_source/chain-0.csv'],config['source_hashes']['new_source/chain-1.csv']=config['source_hashes']['new_source/chain-1.csv'],config['source_hashes']['new_source/chain-0.csv']
    if change=='data':(tmp_path/'new_source/data.json').write_text('{"data":2}')
    if change=='missing_chunk':(old/'run-2-chunk-000000.npz').unlink()
    with pytest.raises(ValueError):reuse_reference_chains(tmp_path,old,new,config)
    assert not (new/'chain-reuse.json').exists()
