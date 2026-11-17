"""Bounded two-field joint sampling; stage06 has its own selection contract."""
from .components import initial_coordinates
from dataclasses import dataclass,asdict
from pathlib import Path
import json,hashlib,time,fcntl
import numpy as np
import torch,xarray as xr
from scipy.optimize import minimize
from affine.torch_stream import save_state,load_state
from .sampling import _save_prepared,_load_prepared,_source_identity,_runtime,_verify_config,_threads,_sync
from .enso_lag import MonthlyENSO,ENSOLagTarget
from .lag_blocks import initialize_lag_blocks,advance_lag_blocks
from .spatial import GaussianSpatialPrior
from .validation_scope import require_laptop_scope


def predictor_hash(predictor):
    h=hashlib.sha256(json.dumps(dict(upper_lag=predictor.upper_lag,origin_month=predictor.origin_month,anchor=predictor.observation_anchor_ns),sort_keys=True).encode())
    for x in (predictor.source_month,predictor.values,predictor.observation_month):
        h.update(str(x.shape).encode());h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def lag_config_hash(prepared,sigma_prior,lag_prior,predictor):
    return hashlib.sha256(json.dumps(dict(prepared=prepared.checksum,sigma=asdict(sigma_prior),lag=asdict(lag_prior),predictor=predictor_hash(predictor)),sort_keys=True).encode()).hexdigest()


def _sources():
    d=_source_identity()
    for name in ('enso_lag.py','lag_sampling.py','lag_blocks.py','spatial.py','api.py','input_contract.py','__init__.py','compute.py','structured_mps.py'):
        d[name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    return d


def _save_predictor(folder,predictor):
    meta=dict(upper_lag=predictor.upper_lag,anchor=predictor.observation_anchor_ns)
    np.savez(folder/'predictor.npz',metadata=json.dumps(meta),source_month=predictor.source_month+predictor.origin_month,values=predictor.values,observation_month=predictor.observation_month+predictor.origin_month)


def _load_predictor(folder):
    with np.load(folder/'predictor.npz',allow_pickle=False) as z:
        m=json.loads(str(z['metadata']))
        return MonthlyENSO(z['source_month'],z['values'],z['observation_month'],m['upper_lag'],m['anchor'])


@dataclass
class LagFitResult:
    path: Path
    prepared: object
    predictor: MonthlyENSO
    config: dict

    def target(self):
        return ENSOLagTarget(self.prepared,GaussianSpatialPrior(**self.config['sigma_prior']),GaussianSpatialPrior(**self.config['lag_prior']),self.predictor,backend=self.config['backend'])

    def posterior(self):
        rows=[];logs=[];iterations=None;p=self.prepared
        for run in range(self.config['runs']):
            chunks=[];lp=[];ix=[]
            for f in sorted((self.path/f'run-{run}').glob('block-*.npz')):
                with np.load(f) as z:chunks.append(z['draws']);lp.append(z['draw_logp']);ix.append(z['retained_iterations'])
            if not chunks:raise ValueError('No retained lag sampling window')
            current=np.concatenate(ix)
            if iterations is not None and not np.array_equal(iterations,current):raise ValueError('Unequal lag run windows; resume saved states')
            iterations=current;rows.append(np.concatenate(chunks));logs.append(np.concatenate(lp))
        flat=np.stack(rows);values=flat.reshape(*flat.shape[:-1],p.cells,len(p.parameter_names)+1)
        coords=dict(run=np.arange(len(rows)),draw=iterations,walker=np.arange(values.shape[2]),cell=np.arange(p.cells),parameter=[*p.parameter_names,'lag_ENSO'])
        coords.update({k:('cell',v) for k,v in p.cell_coordinates.items()})
        return xr.Dataset(dict(unconstrained=(('run','draw','walker','cell','parameter'),values),joint_log_probability=(('run','draw','walker'),np.stack(logs))),coords=coords,attrs=dict(model=p.model,backend=self.config['backend'],joint=True,walkers_are_independent_chains=False,spatial_cells_are_independent=False,lag_convention='ENSO(t-lag)',diagnostic_status='NOT_YET_DIAGNOSED'))


def _continue(fit,target,end,started,wall_seconds):
    folder=fit.path;c=fit.config;p=fit.prepared
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);states=[]
        for run in range(c['runs']):
            blocks=sorted((folder/f'run-{run}').glob('block-*.npz'))
            states.append(load_state(blocks[-1] if blocks else folder/f'run-{run}/initial.npz',provenance=c['provenance']))
        while min(s.iteration for s in states)<end and time.time()-started<wall_seconds:
            for run,state in enumerate(states):
                if state.iteration>=end:continue
                if time.time()-started>=wall_seconds:break
                count=min(c['chunk_sweeps'],end-state.iteration);_sync(p.y.device);before=time.perf_counter()
                new,draws,lp,extra,steps=advance_lag_blocks(target,state,count,deadline_unix=started+wall_seconds);_sync(p.y.device);elapsed=time.perf_counter()-before
                if new.iteration==state.iteration:break
                dest=folder/f'run-{run}'/f'block-{state.iteration:09d}-{new.iteration:09d}.npz'
                if dest.exists():raise ValueError('Do not overwrite lag checkpoints')
                save_state(dest,new,payload=dict(draws=draws.cpu().numpy(),draw_logp=lp.cpu().numpy(),retained_iterations=steps,sampling_seconds=np.array(elapsed)));states[run]=new
        latest=dict(status='SAMPLING_COMPLETE_NOT_CONVERGENCE' if all(s.iteration>=end for s in states) else 'WALL_CAP_CHECKPOINTED',iterations=[s.iteration for s in states],target_iteration=end,actual_wall_seconds=time.time()-started)
        tmp=folder/'latest.tmp';tmp.write_text(json.dumps(latest,indent=2)+'\n');tmp.replace(folder/'latest.json')
    return fit


def sample_enso_lag(prepared,sigma_prior,lag_prior,predictor,*,output_dir,backend=None,dtype=None,device=None,half_walkers=64,runs=4,warmup=128,retained_sweeps=256,chunk_sweeps=16,seed=2026113101,wall_seconds=600,phase='pilot',selection=None,reference=False,centers=None,initial_halves=None,torch_threads=1,numba_threads=8):
    """Two Gaussian fields with independent priors; explicit stage06 workload.

    Default centers optimize the full joint target. For a multimodal investigation
    callers may provide one center per independent run; every center is optimized
    separately and each resulting cloud is full-rank in all actual moved blocks.
    """
    from .input_contract import validate_prepared_seasonality
    validate_prepared_seasonality(prepared)
    started=time.time();p=prepared;folder=Path(output_dir).resolve();dimension=p.cells*(len(p.parameter_names)+1)
    ids=tuple(map(str,p.cell_coordinates.get('panel',np.arange(p.cells))))
    h=lag_config_hash(p,sigma_prior,lag_prior,predictor)
    require_laptop_scope(ids,stage='06',reference=reference,phase=phase,selection=selection,config_hash=h)
    if phase=='pilot' and not reference and wall_seconds>600:raise ValueError('Preselection lag pilot cap is600actualseconds')
    if half_walkers<max(p.cells,len(p.parameter_names)-1)+1 or runs<4 or min(retained_sweeps,chunk_sweeps)<1 or chunk_sweeps>50 or warmup<0 or wall_seconds<=0:raise ValueError('Full-rank moved blocks,4runs and bounded positive budgets required')
    if folder.exists():raise ValueError('Preserve lag fit; use resume_enso_lag')
    from .compute import prepared_for_backend, select_backend
    backend = select_backend(backend,dtype=dtype,device=device)
    p=prepared_for_backend(p,backend,dtype=dtype,device=device)
    select_backend(backend,dtype=p.y.dtype,device=p.y.device,announce=True)
    from .prior_specification import announce_model
    announce_model(p.model,p.regressor_names,p.priors,panels=p.cells,fields={'sigma_trend':sigma_prior,'ENSO_lag':lag_prior})
    with _threads(backend,torch_threads,numba_threads):
        target=ENSOLagTarget(p,sigma_prior,lag_prior,predictor,backend=backend)
        base=list(initial_coordinates(p.model,p.regressors.shape[-1]))
        default=np.tile([*base,np.log(2/(lag_prior.upper-2))] if lag_prior.upper>2 else [*base,0.],p.cells)
        initial=np.asarray(default if centers is None else centers,dtype=float)
        if initial.size==dimension:initial=initial.reshape(1,dimension)
        if initial.shape not in ((1,dimension),(runs,dimension)):raise ValueError('One common or one per-run full joint center required')
        dummy=p.y.new_tensor(initial[0]).expand(half_walkers,-1).clone();u=dummy.reshape(half_walkers,p.cells,target.parameters);compile_info={}
        if backend=='numba':
            from .numba_prototype import compile_specialization
            compile_info=compile_specialization(u[...,:target.base_parameters],p.y,p.variance,p.observed,target.regressors(u),p.model,algorithm='structured',research_float32=True,S=p.priors['S'],sampling=p.sampling)
        elif backend=='mps':
            from .structured_mps import compile_kernel
            compile_info['compilation_or_cache_load_seconds']=compile_kernel(p.model,p.regressors.shape[-1])[1]
        before=time.perf_counter();target(dummy);target(dummy[:1]);_sync(p.y.device);compile_info['disposable_exact_shape_warmup_seconds']=time.perf_counter()-before
        before=time.perf_counter();optimization=[]
        if initial_halves is None:
            def objective(z):
                if time.time()-started>=wall_seconds:raise TimeoutError('Lag optimization reached wall cap')
                with torch.no_grad():return -float(target(p.y.new_tensor(z).reshape(1,-1))[0].cpu())
            optimized=[]
            for center in initial:
                result=minimize(objective,center,method='Powell',options=dict(maxiter=60,maxfev=3000,xtol=1e-4,ftol=1e-6))
                if not np.isfinite(result.fun):raise ValueError('No finite lag initialization point')
                optimized.append(result.x);optimization.append(dict(success=bool(result.success),evaluations=result.nfev,objective=result.fun,center=result.x.tolist()))
            halves=[]
            for run in range(runs):
                rng=np.random.default_rng(seed+run);center=optimized[0 if len(optimized)==1 else run]
                halves.append(tuple(p.y.new_tensor(center+.01*rng.normal(size=(half_walkers,dimension))) for _ in range(2)))
        else:halves=[tuple(h.to(device=p.y.device,dtype=p.y.dtype) for h in pair) for pair in initial_halves]
        if len(halves)!=runs:raise ValueError('One half-pair per independent run required')
        nuisance=[i for i in range(target.parameters) if i not in target.field_indices]
        for pair in halves:
            if len(pair)!=2:raise ValueError('Two complementary halves required')
            for half in pair:
                if half.shape!=(half_walkers,dimension):raise ValueError('Wrong joint cloud shape')
                shaped=half.cpu().numpy().reshape(half_walkers,p.cells,target.parameters)
                blocks=[shaped[...,j] for j in target.field_indices]+[shaped[:,g,nuisance] for g in range(p.cells)]
                if not all(np.linalg.matrix_rank(b-b.mean(0))==b.shape[-1] for b in blocks) or not bool(torch.isfinite(target(half)).all()):raise ValueError('Rank-deficient or unsupported lag cloud')
        initialization=dict(method='Optimized transformed joint centers;0.01 absolute transformed-coordinate spread',optimization=optimization,seconds=time.perf_counter()-before)
        folder.mkdir(parents=True);_save_prepared(folder,p);_save_predictor(folder,predictor)
        c=dict(schema=1,kind='joint_sigma_trend_enso_lag',backend=backend,runs=runs,half_walkers=half_walkers,warmup=warmup,thin=1,chunk_sweeps=chunk_sweeps,seed=seed,sigma_prior=asdict(sigma_prior),lag_prior=asdict(lag_prior),source_identity=_sources(),runtime=_runtime(backend),threads=dict(torch=torch_threads,numba=numba_threads),prepared_checksum=p.checksum,prepared_file_sha256=hashlib.sha256((folder/'prepared.npz').read_bytes()).hexdigest(),predictor_file_sha256=hashlib.sha256((folder/'predictor.npz').read_bytes()).hexdigest(),scope=dict(stage='06',phase=phase,reference=reference,selection=selection,config_hash=h),moved_dimension=max(p.cells,len(nuisance)),total_parameter_dimension=dimension,update_schedule='sigma_field_lag_field_local',target_evaluations_per_sweep=6,original_started_unix=started,original_wall_seconds=wall_seconds)
        c['provenance']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        for name,value in [('config',c),('initialization',initialization),('compilation',compile_info)]:
            (folder/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        for run in range(runs):
            state=initialize_lag_blocks(target,halves[run],seed=seed+10000+run,warmup=warmup,thin=1,provenance=c['provenance'])
            save_state(folder/f'run-{run}/initial.npz',state)
        return _continue(LagFitResult(folder,p,predictor,c),target,warmup+retained_sweeps,started,wall_seconds)


def load_enso_lag(output_dir):
    folder=Path(output_dir).resolve();c=json.loads((folder/'config.json').read_text());_verify_config(c)
    if c['kind']!='joint_sigma_trend_enso_lag':raise ValueError('Not an ENSO-lag fit')
    for file,key in [('prepared.npz','prepared_file_sha256'),('predictor.npz','predictor_file_sha256')]:
        if hashlib.sha256((folder/file).read_bytes()).hexdigest()!=c[key]:raise ValueError('Saved input changed')
    return LagFitResult(folder,_load_prepared(folder),_load_predictor(folder),c)


def resume_enso_lag(output_dir,*,retained_sweeps,wall_seconds=600):
    started=time.time();fit=load_enso_lag(output_dir);c=fit.config;p=fit.prepared
    wall_seconds=min(wall_seconds,c['original_wall_seconds']-(started-c['original_started_unix']))
    if wall_seconds<=0:raise ValueError('Original lag job cap exhausted; no silent reset')
    if c['source_identity']!=_sources() or c['runtime']!=_runtime(c['backend']):raise ValueError('Lag source/runtime changed; explicit migration required')
    ids=tuple(map(str,p.cell_coordinates.get('panel',np.arange(p.cells))));scope=c['scope']
    require_laptop_scope(ids,stage='06',reference=scope['reference'],phase=scope['phase'],selection=scope['selection'],config_hash=scope['config_hash'])
    with _threads(c['backend'],c['threads']['torch'],c['threads']['numba']):
        target=fit.target();initial=load_state(fit.path/'run-0/initial.npz',provenance=c['provenance'])
        before=time.perf_counter();target(initial.halves[0]);_sync(p.y.device)
        (fit.path/f'resume-warmup-{time.time_ns()}.json').write_text(json.dumps(dict(compile_cache_and_disposable_warmup_seconds=time.perf_counter()-before))+'\n')
        return _continue(fit,target,c['warmup']+retained_sweeps,started,wall_seconds)
