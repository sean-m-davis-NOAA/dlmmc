"""Fully joint affine moves, bounded checkpoints and explicit spatial scope."""
from .components import initial_coordinates
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,time,fcntl
import numpy as np
import torch,xarray as xr
from scipy.optimize import minimize
from affine.torch_stream import initialize,advance,save_state,load_state
from .spatial import SpatialTarget,GaussianSpatialPrior
from .sampling import _save_prepared,_load_prepared,_source_identity,_runtime,_verify_config,_threads,_sync
from .validation_scope import require_laptop_scope


def spatial_config_hash(prepared,prior):
    return hashlib.sha256(json.dumps(dict(prepared=prepared.checksum,prior=asdict(prior)),sort_keys=True).encode()).hexdigest()


def _sources():
    sources=_source_identity()
    for name in ('spatial.py','spatial_sampling.py','spatial_blocks.py','api.py','input_contract.py','__init__.py','compute.py','structured_mps.py'):
        sources[name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    return sources


def _panel_ids(p):
    if 'panel' in p.cell_coordinates:return tuple(map(str,p.cell_coordinates['panel']))
    return tuple('|'.join(f'{k}={v[i]}' for k,v in p.cell_coordinates.items()) for i in range(p.cells))


@dataclass
class SpatialFitResult:
    path: Path
    prepared: object
    config: dict

    def posterior(self):
        rows=[];logs=[];iterations=None;p=self.prepared
        for run in range(self.config['runs']):
            chunks=[];lp=[];ix=[]
            for f in sorted((self.path/f'run-{run}').glob('block-*.npz')):
                with np.load(f) as z:
                    chunks.append(z['draws']);lp.append(z['draw_logp']);ix.append(z['retained_iterations'])
            current=np.concatenate(ix)
            if iterations is not None and not np.array_equal(iterations,current):raise ValueError('Unequal run windows; resume the saved joint fit')
            iterations=current;rows.append(np.concatenate(chunks));logs.append(np.concatenate(lp))
        flat=np.stack(rows);values=flat.reshape(*flat.shape[:-1],p.cells,len(p.parameter_names))
        coords=dict(run=np.arange(len(rows)),draw=iterations,walker=np.arange(values.shape[2]),cell=np.arange(p.cells),parameter=p.parameter_names)
        coords.update({k:('cell',v) for k,v in p.cell_coordinates.items()})
        return xr.Dataset(dict(unconstrained=(('run','draw','walker','cell','parameter'),values),joint_log_probability=(('run','draw','walker'),np.stack(logs))),coords=coords,
            attrs=dict(model=p.model,backend=self.config['backend'],joint=True,walkers_are_independent_chains=False,spatial_cells_are_independent=False,diagnostic_status='NOT_YET_DIAGNOSED'))


def _continue(folder,p,c,target,end,started,wall_seconds):
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        states=[]
        for run in range(c['runs']):
            blocks=sorted((folder/f'run-{run}').glob('block-*.npz'))
            states.append(load_state(blocks[-1] if blocks else folder/f'run-{run}'/'initial.npz',provenance=c['provenance']))
        while min(s.iteration for s in states)<end and time.time()-started<wall_seconds:
            for run,state in enumerate(states):
                if state.iteration>=end:continue
                if time.time()-started>=wall_seconds:break
                count=min(c['chunk_sweeps'],end-state.iteration)
                _sync(p.y.device);before=time.perf_counter()
                if c.get('update_schedule','global')=='field_local':
                    from .spatial_blocks import advance_blocks
                    new,draws,lp,extras,steps=advance_blocks(target,state,count,deadline_unix=started+wall_seconds)
                else:new,draws,lp,extras,steps=advance(target,state,count)
                _sync(p.y.device)
                elapsed=time.perf_counter()-before
                if new.iteration==state.iteration:break
                dest=folder/f'run-{run}'/f'block-{state.iteration:09d}-{new.iteration:09d}.npz'
                if dest.exists():raise ValueError('Cannot overwrite joint checkpoint')
                save_state(dest,new,payload=dict(draws=draws.cpu().numpy(),draw_logp=lp.cpu().numpy(),retained_iterations=steps,sampling_seconds=np.array(elapsed)))
                states[run]=new
        latest=dict(status='SAMPLING_COMPLETE_NOT_CONVERGENCE' if all(s.iteration>=end for s in states) else 'WALL_CAP_CHECKPOINTED',iterations=[s.iteration for s in states],target_iteration=end,actual_wall_seconds=time.time()-started)
        temp=folder/'latest.tmp';temp.write_text(json.dumps(latest,indent=2)+'\n');temp.replace(folder/'latest.json')
    return SpatialFitResult(folder,p,c)


def sample_spatial(prepared,prior,*,output_dir,backend='numba',half_walkers=64,runs=4,warmup=128,retained_sweeps=256,chunk_sweeps=16,seed=2026112701,wall_seconds=600,phase='pilot',selection=None,reference=False,centers=None,initial_halves=None,torch_threads=1,numba_threads=8,update_schedule='global'):
    """Stage05 joint fit. Post-pilot calls require a matching explicit selection.

    ``prior`` is GaussianSpatialPrior on sigma_trend in normalized model units.
    Complete joint draw alignment is retained for conditional reconstruction.
    No-seasonal models require a durable declaration from deseasonalized preparation.
    """
    from .input_contract import validate_prepared_seasonality
    validate_prepared_seasonality(prepared)
    started=time.time();p=prepared;folder=Path(output_dir).resolve();dimension=p.cells*len(p.parameter_names)
    if phase=='pilot' and not reference and wall_seconds>600:raise ValueError('Preselection pilot cap is600actualseconds')
    if update_schedule not in ('global','field_local'):raise ValueError('Unknown joint update schedule')
    moved_dimension=dimension if update_schedule=='global' else max(p.cells,len(p.parameter_names)-1)
    config_hash=spatial_config_hash(p,prior)
    require_laptop_scope(_panel_ids(p),stage='05',reference=reference,phase=phase,selection=selection,config_hash=config_hash)
    if half_walkers<moved_dimension+1 or runs<4 or min(retained_sweeps,chunk_sweeps)<1 or chunk_sweeps>50 or warmup<0 or wall_seconds<=0:
        raise ValueError('Full-rank joint halves,4runs,positive budgets and chunks<=50 required')
    if folder.exists():raise ValueError('Preserve existing joint fit; use resume_spatial')
    from .compute import prepared_for_backend
    p=prepared_for_backend(p,backend)
    with _threads(backend,torch_threads,numba_threads):
        target=SpatialTarget(p,prior,backend=backend)
        default=initial_coordinates(p.model,p.regressors.shape[-1])
        center=np.broadcast_to(default,(p.cells,len(default))).copy().ravel() if centers is None else np.asarray(centers).reshape(dimension).copy()
        dummy=p.y.new_tensor(center).expand(half_walkers,-1).clone();compile_info={}
        if backend=='numba':
            from .numba_prototype import compile_specialization
            compile_info=compile_specialization(dummy.reshape(half_walkers,p.cells,-1),*p.arguments,p.model,algorithm='structured',S=p.priors['S'],sampling=p.sampling)
        elif backend=='mps':
            from .structured_mps import compile_kernel
            compile_info['compilation_or_cache_load_seconds']=compile_kernel(p.model,p.regressors.shape[-1])[1]
        _sync(p.y.device);before=time.perf_counter();target(dummy);target(dummy[:1]);_sync(p.y.device)
        compile_info['disposable_exact_shape_warmup_seconds']=time.perf_counter()-before
        before=time.perf_counter()
        if initial_halves is None:
            def objective(z):
                if time.time()-started>=wall_seconds:raise TimeoutError('Joint initialization reached actual wall cap')
                with torch.no_grad():return -float(target(p.y.new_tensor(z).reshape(1,-1))[0].cpu())
            opt=minimize(objective,center,method='Powell',options=dict(maxiter=60,maxfev=2000,xtol=1e-4,ftol=1e-6))
            if not np.isfinite(opt.fun):raise ValueError('No supported joint center')
            center=opt.x;halves=[]
            for run in range(runs):
                rng=np.random.default_rng(seed+run)
                halves.append(tuple(p.y.new_tensor(center+.01*rng.normal(size=(half_walkers,dimension))) for _ in range(2)))
            initialization=dict(method='Joint optimized transformed point;0.01 absolute transformed-coordinate scales',center=center.tolist(),success=bool(opt.success),evaluations=opt.nfev,objective=opt.fun)
        else:
            halves=[tuple(h.to(device=p.y.device,dtype=p.y.dtype) for h in pair) for pair in initial_halves];initialization=dict(method='Explicit validated half clouds')
        if len(halves)!=runs:raise ValueError('One independent half-pair per run required')
        for pair in halves:
            if len(pair)!=2:raise ValueError('Two complementary halves required')
            for half in pair:
                if half.shape!=(half_walkers,dimension):raise ValueError('Joint halves must be rank2 and include full moved dimension')
                z=half.cpu().numpy()
                if update_schedule=='global':rank_ok=np.linalg.matrix_rank(z-z.mean(0))==dimension
                else:
                    shaped=z.reshape(half_walkers,p.cells,len(p.parameter_names));field=shaped[...,target.index];nuisance=[i for i in range(target.parameters) if i!=target.index]
                    rank_ok=np.linalg.matrix_rank(field-field.mean(0))==p.cells and all(np.linalg.matrix_rank(shaped[:,g,nuisance]-shaped[:,g,nuisance].mean(0))==len(nuisance) for g in range(p.cells))
                if not rank_ok or not bool(torch.isfinite(target(half)).all()):raise ValueError('Rank-deficient or unsupported initial cloud')
        initialization['seconds']=time.perf_counter()-before
        folder.mkdir(parents=True);_save_prepared(folder,p)
        c=dict(schema=1,kind='joint_sigma_trend',backend=backend,runs=runs,half_walkers=half_walkers,warmup=warmup,thin=1,chunk_sweeps=chunk_sweeps,seed=seed,prior=asdict(prior),source_identity=_sources(),runtime=_runtime(backend),threads=dict(torch=torch_threads,numba=numba_threads),prepared_checksum=p.checksum,prepared_file_sha256=hashlib.sha256((folder/'prepared.npz').read_bytes()).hexdigest(),scope=dict(stage='05',phase=phase,reference=reference,selection=selection,config_hash=config_hash),moved_dimension=moved_dimension,total_parameter_dimension=dimension,update_schedule=update_schedule,original_started_unix=started,original_wall_seconds=wall_seconds)
        c['provenance']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        for name,value in [('config',c),('initialization',initialization),('compilation',compile_info)]:
            (folder/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        for run in range(runs):
            if update_schedule=='field_local':
                from .spatial_blocks import initialize_blocks
                state=initialize_blocks(target,halves[run],seed=seed+10000+run,warmup=warmup,thin=1,provenance=c['provenance'])
            else:state=initialize(target,halves[run],seed=seed+10000+run,warmup=warmup,thin=1,provenance=c['provenance'])
            save_state(folder/f'run-{run}'/'initial.npz',state)
        return _continue(folder,p,c,target,warmup+retained_sweeps,started,wall_seconds)


def load_spatial(output_dir):
    folder=Path(output_dir).resolve();c=json.loads((folder/'config.json').read_text());_verify_config(c)
    if c['kind']!='joint_sigma_trend':raise ValueError('Not a stage05 joint fit')
    if hashlib.sha256((folder/'prepared.npz').read_bytes()).hexdigest()!=c['prepared_file_sha256']:raise ValueError('Prepared input changed')
    return SpatialFitResult(folder,_load_prepared(folder),c)


def resume_spatial(output_dir,*,retained_sweeps,wall_seconds=600):
    started=time.time();fit=load_spatial(output_dir);c=fit.config;p=fit.prepared
    wall_seconds=min(wall_seconds,c['original_wall_seconds']-(started-c['original_started_unix']))
    if wall_seconds<=0:raise ValueError('Original joint job cap exhausted; no silent budget reset')
    if c['source_identity']!=_sources() or c['runtime']!=_runtime(c['backend']):raise ValueError('Joint source/runtime changed; explicit migration required')
    scope=c['scope'];require_laptop_scope(_panel_ids(p),stage='05',reference=scope['reference'],phase=scope['phase'],selection=scope['selection'],config_hash=scope['config_hash'])
    with _threads(c['backend'],c['threads']['torch'],c['threads']['numba']):
        target=SpatialTarget(p,GaussianSpatialPrior(**c['prior']),backend=c['backend'])
        initial=load_state(fit.path/'run-0/initial.npz',provenance=c['provenance'])
        before=time.perf_counter();target(initial.halves[0]);_sync(p.y.device)
        (fit.path/f'resume-warmup-{time.time_ns()}.json').write_text(json.dumps(dict(compile_cache_and_disposable_warmup_seconds=time.perf_counter()-before))+'\n')
        return _continue(fit.path,p,c,target,c['warmup']+retained_sweeps,started,wall_seconds)
