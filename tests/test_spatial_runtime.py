from dlmmc.spatial_runtime import joint_affine_resources


def test_resource_projection_includes_actual_joint_summary_axis():
    small=joint_affine_resources(4,396,12,update_schedule='field_local')
    assert small['joint_summary_quantities']==32
    assert small['half_walkers']==64
    assert small['joint_summary_window_bytes']==4*4000*128*32*8
    full=joint_affine_resources(324,396,12,update_schedule='field_local')
    assert full['half_walkers']==512
    assert full['joint_summary_quantities']==106272
    assert full['diagnostic_working_bytes_lower_bound']>10**13
    assert full['target_evaluations_per_sweep']==4
