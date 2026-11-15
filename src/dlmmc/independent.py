"""Continue an explicitly selected independent-cell subset without refitting."""
from dataclasses import replace
from pathlib import Path
import hashlib,json,fcntl
import numpy as np
from affine.torch_stream import load_state,save_state
from .sampling import load_fit,_source_identity,_save_prepared,FitResult,make_target,_threads


def fork_cells(fit, cells, *, output_dir):
    """Copy selected cells and complete checkpoints into a new continuation.

    ``cells`` are integer positions in the fit's recorded cell coordinate.
    All past draws, global retention phase, both halves and per-run RNG states
    are retained. Resuming a smaller batch changes subsequent random-call
    shapes: it is a valid independent-target continuation, not a promise of
    identical future draws to the original full batch. The fork is not an
    independent replicate of its parent and must never be compared as one.
    Coupled fits do not use this interface.
    """
    cells=list(cells)
    if not cells or len(set(cells))!=len(cells) or any(not isinstance(c,(int,np.integer)) or not 0<=c<fit.prepared.cells for c in cells):
        raise ValueError('Nonempty unique in-range integer cell positions required')
    if fit.config.get('joint',False):raise ValueError('Coupled cells cannot be forked independently')
    if fit.config['source_identity']!=_source_identity():raise ValueError('Source changed; continuation needs explicit migration')
    destination=Path(output_dir).resolve()
    if destination.exists():raise ValueError('Use a new output directory; parent is retained')
    with (fit.path/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=load_fit(fit.path);p=source.prepared
        checksum=hashlib.sha256(json.dumps(dict(parent=p.checksum,cells=cells),sort_keys=True).encode()).hexdigest()
        subset=replace(p,y=p.y[cells].clone(),variance=p.variance[cells].clone(),observed=p.observed[cells].clone(),
            regressors=p.regressors[cells].clone(),offset=p.offset[cells].copy(),scale=p.scale[cells].copy(),
            cell_coordinates={k:v[cells].copy() for k,v in p.cell_coordinates.items()},checksum=checksum)
        destination.mkdir(parents=True)
        _save_prepared(destination,subset)
        config={k:v for k,v in source.config.items() if k!='provenance'}
        config.update(prepared_checksum=checksum,prepared_file_sha256=hashlib.sha256((destination/'prepared.npz').read_bytes()).hexdigest(),
            lineage=dict(parent_directory=str(source.path),parent_provenance=source.config['provenance'],parent_cells=cells,
                         independent_replicate=False,future_full_batch_bitwise_identity=False))
        config['provenance']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        target=make_target(subset,config['backend']);copied=[]
        for run in range(config['runs']):
            files=[source.path/f'run-{run}'/'initial.npz']+sorted((source.path/f'run-{run}').glob('block-*.npz'))
            for path in files:
                state=load_state(path,provenance=source.config['provenance'])
                new=replace(state,halves=tuple(h[:,cells].clone() for h in state.halves),
                    logp=tuple(h[:,cells].clone() for h in state.logp),
                    extras=None if state.extras is None else tuple(h[:,cells].clone() for h in state.extras),
                    accepted=state.accepted[:,:,cells].clone(),proposed=state.proposed[:,:,cells].clone(),
                    warmup_accepted=state.warmup_accepted[:,:,cells].clone(),warmup_proposed=state.warmup_proposed[:,:,cells].clone(),
                    provenance=config['provenance'])
                # Check the target on each terminal state: a subset must not
                # alter per-cell normalization, priors or density.
                if path==files[-1]:
                    import torch
                    with _threads(config['backend'],config['threads']['torch'],config['threads']['numba']):
                        for h,lp in zip(new.halves,new.logp):
                            torch.testing.assert_close(target(h),lp,atol=1e-8,rtol=0)
                payload=None
                if path.name!='initial.npz':
                    with np.load(path,allow_pickle=False) as z:
                        payload={'draws':z['draws'][:,:,cells], 'draw_logp':z['draw_logp'][:,:,cells],
                                 'retained_iterations':z['retained_iterations'],
                                 'parent_sampling_seconds':z['sampling_seconds'] if 'sampling_seconds' in z else z['parent_sampling_seconds']}
                save_state(destination/f'run-{run}'/path.name,new,payload=payload)
                copied.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (destination/'lineage.json').write_text(json.dumps(dict(source_chunks=copied,scope='Inherited samples are not independent new evidence'),indent=2)+'\n')
        (destination/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (destination/'latest.json').write_text(json.dumps(dict(status='INDEPENDENT_SUBSET_FORKED_NOT_NEW_EVIDENCE',parent=str(source.path),cells=cells),indent=2)+'\n')
    return FitResult(destination,subset,config)
