from copy import deepcopy
import pytest
from dlmmc.validation_coverage_refinement import endpoint_alpha, merge_coverage, WINDOWS


def baseline():
    names = ['sigma_trend','sigma_seas','sigma_AR','rhoAR1']
    datasets = [dict(replication=r,status='ANALYZED_NOT_MODEL_QUALIFICATION',
                     quantities=[dict(name=n,rank=r%5,exact_ties=0,coverage=dict(
                         membership='INCONCLUSIVE' if r==18 and n=='sigma_seas' else 'COVERED'))
                                 for n in names]) for r in range(20)]
    return dict(model='vanilla_ar1',retained_steps=16000,datasets=datasets,
                aggregate=[dict(name=n,rank_counts=[4]*5,rank_pvalue=1.,rank_status='PASS',
                                coverage=dict(status='INCONCLUSIVE' if n=='sigma_seas' else 'PASS')) for n in names])


def test_spending_continues_without_reset():
    assert [endpoint_alpha(n) for n in WINDOWS] == [.05/(k*(k+1))/1040 for k in range(3,7)]
    with pytest.raises(ValueError): endpoint_alpha(16000)
    with pytest.raises(ValueError): endpoint_alpha(512000)


@pytest.mark.parametrize('membership,status',[('COVERED','PASS'),('NOT_COVERED','PASS'),('INCONCLUSIVE','INCONCLUSIVE')])
def test_only_approved_coverage_changes(membership,status):
    old=baseline(); saved=deepcopy(old)
    new=merge_coverage(old,dict(membership=membership,endpoint_alpha=endpoint_alpha(32000)),retained=32000)
    assert old==saved
    assert len(new['datasets'])==20 and new['status']==status
    for r in range(20):
        for j in range(4):
            q=new['datasets'][r]['quantities'][j]; original=old['datasets'][r]['quantities'][j]
            if (r,j)!=(18,1): assert q==original
            assert (q['rank'],q['exact_ties'])==(original['rank'],original['exact_ties'])
    for q,original in zip(new['aggregate'],old['aggregate']):
        assert {k:v for k,v in q.items() if k!='coverage'}=={k:v for k,v in original.items() if k!='coverage'}


@pytest.mark.parametrize('mutation',['model','missing_dataset','already_classified','alpha'])
def test_rejects_unapproved_scope(mutation):
    old=baseline(); coverage=dict(membership='COVERED',endpoint_alpha=endpoint_alpha(32000))
    if mutation=='model': old['model']='ar2'
    if mutation=='missing_dataset': old['datasets'].pop()
    if mutation=='already_classified': old['datasets'][18]['quantities'][1]['coverage']['membership']='COVERED'
    if mutation=='alpha': coverage['endpoint_alpha']=.05
    with pytest.raises(ValueError): merge_coverage(old,coverage,retained=32000)
