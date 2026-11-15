"""Analytical resource accounting; does not allocate or fit the requested grid."""
from .tuning import walker_candidates


def joint_affine_resources(cells,time_points,state_dimension,*,parameters_per_cell=4,retained_sweeps=4000,runs=4,chunk_sweeps=50,bytes_per_value=8,update_schedule='global'):
    if min(cells,time_points,state_dimension,parameters_per_cell,retained_sweeps,runs,chunk_sweeps,bytes_per_value)<1:
        raise ValueError('Positive explicit dimensions and budgets required')
    if update_schedule not in ('global','field_local'):raise ValueError('Unknown schedule')
    dimension=cells*parameters_per_cell;moved=dimension if update_schedule=='global' else max(cells,parameters_per_cell-1)
    half=walker_candidates(moved)[0];walkers=2*half
    # Current implementation builds F for every proposed walker/cell. Sampling
    # checkpoints are bounded, but posterior diagnostics materialize the window.
    f=half*cells*time_points*state_dimension*bytes_per_value
    matrices=4*half*cells*state_dimension**2*bytes_per_value
    ensemble=runs*walkers*dimension*bytes_per_value
    chunk=chunk_sweeps*walkers*dimension*bytes_per_value
    chain=runs*retained_sweeps*walkers*dimension*bytes_per_value
    factors=3*cells*cells*bytes_per_value
    # spatial_summaries materializes physical parameters (including log AR
    # variance), all pair contrasts, and all pair products. Include this actual
    # O(G^2) summary axis rather than scaling only the parameter chain.
    summaries=cells*(parameters_per_cell+1)+cells*(cells-1)
    summary_window=runs*retained_sweeps*walkers*summaries*bytes_per_value
    return dict(cells=cells,moved_dimension=moved,total_parameter_dimension=dimension,update_schedule=update_schedule,target_evaluations_per_sweep=2 if update_schedule=='global' else 4,half_walkers=half,total_walkers=walkers,
        observation_vector_bytes=f,parameter_matrices_bytes=matrices,ensemble_bytes=ensemble,
        parameter_chunk_bytes=chunk,parameter_chain_bytes=chain,prior_factors_bytes=factors,
        sampling_working_bytes_lower_bound=f+matrices+ensemble+chunk+factors,
        joint_summary_quantities=summaries,joint_summary_window_bytes=summary_window,
        diagnostic_working_bytes_lower_bound=chain+summary_window,
        diagnostic_multiplier_scenarios=[4,12],
        prior_setup_complexity='dense O(cells^3); no full-grid factorization measured',
        scope='Analytical dimensions only; allocator/workspace/process overhead and log-probability files additional; not a measured memory maximum')
