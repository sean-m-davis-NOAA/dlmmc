"""Fail-closed workload limits for the laptop validation harness."""

def require_laptop_scope(panel_ids, *, stage='independent', reference=False, selection=None, config_hash=None, phase='pilot'):
    ids=tuple(panel_ids)
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Nonempty unique panel IDs required')
    if reference and stage in ('05','06'):
        if len(ids)>4: raise ValueError('Joint reference capped at four cells')
        return
    if stage not in ('05','06'):
        if len(ids)>5: raise ValueError('Independent validation capped at five cells')
        return
    if phase not in ('pilot', 'selected'):
        raise ValueError('Unknown joint execution phase')
    if phase == 'selected' and selection is None:
        raise ValueError('Post-pilot fit requires explicit selection even for four panels')
    if selection is None:
        if len(ids)>4: raise ValueError('Preselection joint pilot capped at four cells')
        return
    if (selection.get('stage')!=stage or not selection.get('explicit_user_response')
        or not selection.get('estimate_presented') or not config_hash
        or selection.get('config_hash')!=config_hash
        or tuple(selection.get('panel_ids',()))!=ids
        or len(ids)>selection.get('approved_maximum_panels',0)):
        raise ValueError('Missing or mismatched stage-specific user selection')
