from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from affine.torch_stream import initialize,advance,save_state,load_state
from dlmmc import prepare_data
from dlmmc.examples import load_panels
from dlmmc.profiles import LEGACY_PRODUCTION_PROFILE as PRODUCTION_PROFILE
from dlmmc.structured_mps_sampling import validate_scope,gpu_prepared
from dlmmc.structured_mps import StructuredMPSTarget


def prepared():
    y,s,x,_=load_panels(Path(__file__).resolve().parents[1]/'examples/modernization')
    return prepare_data(y,x,uncertainty=s)


def test_research_scope_is_explicit():
    p=prepared();profile=PRODUCTION_PROFILE
    with pytest.raises(ValueError,match='selection'):validate_scope(p,None,profile)
    selection=dict(scope='structured_mps_research_fit',explicit_user_response='named GPU test',
        backend='structured_mps',dtype='float32',profile=profile.to_dict(),panel_ids=list(p.cell_coordinates['panel']))
    validate_scope(p,selection,profile)
    with pytest.raises(ValueError,match='selection'):validate_scope(p,dict(selection,panel_ids=selection['panel_ids'][::-1]),profile)
    with pytest.raises(ValueError,match='settings'):validate_scope(p,selection,replace(profile,retained_sweeps=256))


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Native MPS unavailable')
def test_native_gpu_complete_checkpoint_replay(tmp_path):
    torch.set_num_threads(1);p=gpu_prepared(prepared());target=StructuredMPSTarget(p)
    rng=np.random.default_rng(1501)
    halves=tuple(torch.tensor(np.array([-9.2,-4.6,-2.3,0.])+.01*rng.normal(size=(8,5,4)),
        device='mps',dtype=torch.float32) for _ in range(2))
    with torch.no_grad():
        state=initialize(target,halves,seed=1501,warmup=1,provenance='replay')
        full,draws,lp,_,steps=advance(target,state,4)
        mid,d1,l1,_,s1=advance(target,state,2)
        save_state(tmp_path/'state.npz',mid)
        restored=load_state(tmp_path/'state.npz',provenance='replay')
        split,d2,l2,_,s2=advance(target,restored,2)
        assert torch.equal(draws,torch.cat((d1,d2))) and torch.equal(lp,torch.cat((l1,l2)))
        np.testing.assert_array_equal(steps,np.concatenate((s1,s2)))
        for a,b in zip(full.halves,split.halves):assert torch.equal(a,b)
        for name in ('rng','accepted','proposed','warmup_accepted','warmup_proposed'):
            assert torch.equal(getattr(full,name),getattr(split,name))
        assert full.iteration==split.iteration==4 and draws.shape==(3,16,5,4)
