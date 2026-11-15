import hashlib,json
import numpy as np
import pytest
from dlmmc.reconstruction import ReconstructionResult
from dlmmc.reconstruction_merge import combine_reconstructions


def fixture(path,seed,shift):
    path.mkdir(); config=dict(posterior_sha256='same',model='synthetic',cells=1,cell_coordinates={},parameter_names=['a'],nreg=0,offset=[0],scale=[1],dtype='float64',device='cpu',source_sha256={},seed=seed,selected_per_run=4)
    (path/'config.json').write_text(json.dumps(config));rows=[]
    for run in range(4):
        file=path/f'{run}.npz';draw=np.arange(4)*2+shift
        np.savez(file,selected=np.arange(4),source_draw=draw,source_walker=np.zeros(4,int),time=[0,1],states=np.full((4,1,2,1),seed+run),unconstrained=draw[:,None,None])
        rows.append(dict(file=file.name,run=run,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    (path/'index.json').write_text(json.dumps(dict(status='RECONSTRUCTION_COMPLETE_OUTPUT_DIAGNOSTICS_PENDING',chunks=rows)))
    return ReconstructionResult(path)


def test_merge_preserves_every_draw_and_run_in_time_order(tmp_path):
    a=fixture(tmp_path/'a',10,1);b=fixture(tmp_path/'b',20,0)
    original=(a.path/'index.json').read_bytes()
    out=combine_reconstructions([a,b],output_dir=tmp_path/'merged',chunk_draws=3)
    idx=json.loads((out.path/'index.json').read_text())
    for r in range(4):
        chunks=[np.load(out.path/row['file']) for row in idx['chunks'] if row['run']==r]
        np.testing.assert_array_equal(np.concatenate([z['source_draw'] for z in chunks]),np.arange(8))
        np.testing.assert_array_equal(np.concatenate([z['states'][:,0,0,0] for z in chunks]),np.tile([20+r,10+r],4))
        assert all(len(z['selected'])<=3 for z in chunks)
    assert (a.path/'index.json').read_bytes()==original
    with pytest.raises(ValueError,match='existing'):combine_reconstructions([a,b],output_dir=out.path)


def test_merge_rejects_changed_target_duplicate_seed_and_corruption(tmp_path):
    a=fixture(tmp_path/'a',10,0);b=fixture(tmp_path/'b',20,1)
    p=b.path/'config.json';c=json.loads(p.read_text());c['posterior_sha256']='different';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='posterior'):combine_reconstructions([a,b],output_dir=tmp_path/'bad')
    c['posterior_sha256']='same';c['seed']=10;p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='distinct'):combine_reconstructions([a,b],output_dir=tmp_path/'bad')
    c['seed']=20;p.write_text(json.dumps(c));(b.path/'0.npz').write_bytes(b'bad')
    with pytest.raises(ValueError,match='changed'):combine_reconstructions([a,b],output_dir=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_merge_rejects_unbalanced_run_sets_before_output(tmp_path):
    a=fixture(tmp_path/'a',10,0);b=fixture(tmp_path/'b',20,1)
    p=b.path/'index.json';d=json.loads(p.read_text());d['chunks']=d['chunks'][:-1];p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Unbalanced'):
        combine_reconstructions([a,b],output_dir=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
