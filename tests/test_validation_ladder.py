import copy
import numpy as np
import pytest
from dlmmc.validation_ladder import check_pair,require_summary_diagnostics


def test_pair_must_represent_same_panel_target_and_independent_runs():
    c=dict(status='WINDOW_DIAGNOSTICS_PASS',model='vanilla_ar1',panel='p05',data_sha256='same',sampling_directory='candidate')
    r=dict(c,sampling_directory='reference')
    check_pair(c,r,[1,2,3,4],[5,6,7,8])
    for key,value in [('panel','p10'),('model','vanilla_ar2'),('data_sha256','changed'),('status','INCONCLUSIVE'),('sampling_directory','candidate')]:
        with pytest.raises(ValueError):check_pair(c,dict(r,**{key:value}),[1,2,3,4],[5,6,7,8])
    with pytest.raises(ValueError):check_pair(c,r,[1,1,2,3],[5,6,7,8])
    with pytest.raises(ValueError):check_pair(c,r,[1,2,3,4],[4,5,6,7])


def test_summary_gate_includes_widths_tails_and_controlling_block_length():
    d=dict(iat_length_reliable=[True,True],rank_rhat_ensemble_means=[1.001,1.002],rank_rhat_ensemble_widths=[1.003,1.004],split_rhat_ensemble_means=[1.001,1.002],iat_by_run=np.full((4,2),20.).tolist())
    d['quantile_indicator_diagnostics']={'0.05':copy.deepcopy(d)}
    require_summary_diagnostics(d,[0,1],100)
    with pytest.raises(ValueError):require_summary_diagnostics(d,[0,1],99)
    for key in ['rank_rhat_ensemble_means','rank_rhat_ensemble_widths']:
        changed=copy.deepcopy(d);changed[key][0]=1.01
        with pytest.raises(ValueError):require_summary_diagnostics(changed,[0,1],100)
    changed=copy.deepcopy(d);changed['quantile_indicator_diagnostics']['0.05']['iat_length_reliable'][1]=False
    with pytest.raises(ValueError):require_summary_diagnostics(changed,[0,1],100)
    changed=copy.deepcopy(d);changed['quantile_indicator_diagnostics']['0.05']['iat_by_run'][3][1]=21
    with pytest.raises(ValueError):require_summary_diagnostics(changed,[0,1],100)
