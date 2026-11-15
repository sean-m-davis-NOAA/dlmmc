import pytest
from dlmmc.validation_scope import require_laptop_scope

def test_full_grid_rejected_before_work():
    with pytest.raises(ValueError,match='five'):
        require_laptop_scope(range(324))
    require_laptop_scope(range(5))

@pytest.mark.parametrize('stage',['05','06'])
def test_separate_gates(stage):
    require_laptop_scope(range(4),stage=stage)
    with pytest.raises(ValueError): require_laptop_scope(range(6),stage=stage)
    selection=dict(stage=stage,explicit_user_response='six panels',estimate_presented='report',config_hash='abc',panel_ids=list(range(6)),approved_maximum_panels=6)
    require_laptop_scope(range(6),stage=stage,selection=selection,config_hash='abc')
    with pytest.raises(ValueError): require_laptop_scope(range(6),stage='06' if stage=='05' else '05',selection=selection,config_hash='abc')
    with pytest.raises(ValueError): require_laptop_scope(range(6),stage=stage,selection=selection,config_hash='changed')
    with pytest.raises(ValueError): require_laptop_scope(range(6),stage=stage,reference=True,selection=selection,config_hash='abc')

@pytest.mark.parametrize('stage',['05','06'])
def test_four_panel_postpilot_requires_selection(stage):
    with pytest.raises(ValueError,match='Post-pilot'):
        require_laptop_scope(range(4),stage=stage,phase='selected')
