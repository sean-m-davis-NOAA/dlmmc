from dlmmc.lag_runtime import lag_affine_resources


def test_two_field_resource_axes_include_actual_diagnostic_quantities():
    small=lag_affine_resources(4,396,12)
    assert small['moved_dimension']==4 and small['total_parameter_dimension']==20
    assert small['target_evaluations_per_sweep']==6
    assert small['joint_summary_quantities']==54
    assert small['total_walkers']==128
    large=lag_affine_resources(324,396,12)
    assert large['half_walkers']==512
    assert large['joint_summary_quantities']==211574
    assert large['diagnostic_working_bytes_lower_bound']>25*1024**4
