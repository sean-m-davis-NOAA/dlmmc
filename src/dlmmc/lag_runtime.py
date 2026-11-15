"""Two-field resource accounting without allocating an unselected grid."""
from .spatial_runtime import joint_affine_resources


def lag_affine_resources(cells,time_points,state_dimension,*,base_parameters=4,retained_sweeps=4000,runs=4,chunk_sweeps=50,bytes_per_value=8):
    # Both fields move G coordinates; local nuisance has Pbase-1 coordinates.
    # The reused one-field helper's Pbase+1 planning floor is conservative for
    # the usual G>=4 case; record the actual moved dimension explicitly.
    from .tuning import walker_candidates
    moved=max(cells,base_parameters-1)
    half=walker_candidates(moved)[0]
    d=joint_affine_resources(cells,time_points,state_dimension,parameters_per_cell=base_parameters+1,retained_sweeps=retained_sweeps,runs=runs,chunk_sweeps=chunk_sweeps,bytes_per_value=bytes_per_value,update_schedule='field_local')
    if d['half_walkers']!=half:
        raise ValueError('Resource accounting needs explicit revised walker floor for these dimensions')
    # Physical parameters include lag and log AR variance. Each field adds
    # a mean plus pair contrasts/products; add one cross-field product/cell.
    summaries=cells*(base_parameters+2)+2+2*cells*(cells-1)+cells
    summary_bytes=runs*retained_sweeps*(2*half)*summaries*bytes_per_value
    extra_factors=d['prior_factors_bytes']
    predictor_work=3*half*cells*time_points*bytes_per_value
    d.update(moved_dimension=moved,update_schedule='sigma_field_lag_field_local',target_evaluations_per_sweep=6,prior_factors_bytes=2*extra_factors,joint_summary_quantities=summaries,joint_summary_window_bytes=summary_bytes,diagnostic_working_bytes_lower_bound=d['parameter_chain_bytes']+summary_bytes,sampling_working_bytes_lower_bound=d['sampling_working_bytes_lower_bound']+extra_factors+predictor_work,interpolation_working_bytes_lower_bound=predictor_work,fields=2)
    return d
