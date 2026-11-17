"""Bounded independent production fits, with visible lightweight diagnostics.

Production is not a qualification certificate. Existing four-run inference,
trajectory reconstruction and five-cell validation guards remain unchanged.
"""
from pathlib import Path
from .input_contract import validate_no_seasonal_input
from dataclasses import replace
import hashlib, json, time
import numpy as np
import torch
from scipy.optimize import minimize
from affine.torch_stream import initialize, save_state, load_state, advance
from .prepare import prepare_data
from .profiles import PRODUCTION_PROFILE, validate_production_scope
from .sampling import (FitResult, make_target, _threads, _source_identity,
    _save_prepared, _runtime, load_fit, _sync, _resume_fit)
from .diagnostics import autocorrelation_time
from .inference import physical_parameters
from .validation_derived import finite_moment_summaries


def _identity():
    return {n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
            for n in ('production.py','profiles.py','compute.py')}


def _cell(p, indices):
    return replace(p, y=p.y[indices], variance=p.variance[indices], observed=p.observed[indices],
        regressors=p.regressors[indices], offset=p.offset[indices], scale=p.scale[indices],
        cell_coordinates={k:v[indices] for k,v in p.cell_coordinates.items()})


def prepare_production(data, regressors=None, *, uncertainty, selection=None,
        profile=PRODUCTION_PROFILE, model='vanilla_ar1', backend='numba', dtype='float64',
        device='cpu', priors=None, sampling='monthly'):
    if tuple(data.dims) != ('time','panel') or str(device) != 'cpu':
        raise ValueError('Production expects named time/panel data on CPU')
    validate_production_scope(data.panel.values,selection,model=model,backend=backend,dtype=dtype,profile=profile)
    from .input_contract import validate_no_seasonal_input
    validate_no_seasonal_input(data,model)
    pieces=[]
    for start in range(0,data.sizes['panel'],5):
        ix=slice(start,start+5)
        pieces.append(prepare_data(data.isel(panel=ix),(regressors.isel(panel=ix) if regressors is not None and "panel" in getattr(regressors, "dims", ()) else regressors),
            uncertainty=uncertainty.isel(panel=ix),model=model,dtype=dtype,device=device,priors=priors,sampling=sampling))
    p=pieces[0]
    digest=hashlib.sha256(json.dumps([x.checksum for x in pieces]).encode()).hexdigest()
    return replace(p,**{n:torch.cat([getattr(x,n) for x in pieces]) for n in ('y','variance','observed','regressors')},
        offset=np.concatenate([x.offset for x in pieces]),scale=np.concatenate([x.scale for x in pieces]),
        cell_coordinates={k:np.concatenate([x.cell_coordinates[k] for x in pieces]) for k in p.cell_coordinates},checksum=digest)


def initial_center(p):
    from .components import initial_coordinates
    return initial_coordinates(p.model,p.regressors.shape[-1])


def fit_production(data, regressors=None, *, uncertainty, output_dir, selection=None,
        profile=PRODUCTION_PROFILE, seed=2026111401, wall_seconds=5400,
        centers=None, model='vanilla_ar1', backend=None, dtype=None,
        device=None, priors=None, sampling='monthly', numba_threads=8):
    from .compute import resolve_compute
    compute=resolve_compute(backend,device=device,dtype=dtype,announce=True)
    backend,dtype,device=compute.backend,compute.dtype,compute.device
    if backend in ('mps','cuda'):
        from .gpu_production import fit_gpu_production
        return fit_gpu_production(data,regressors,uncertainty=uncertainty,output_dir=output_dir,selection=selection,
            profile=profile,seed=seed,wall_seconds=wall_seconds,centers=centers,model=model,backend=backend,
            device=device,priors=priors,sampling=sampling,numba_threads=numba_threads)
    started=time.time();folder=Path(output_dir).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Preserve existing production fit; use explicit resume')
    p=prepare_production(data,regressors,uncertainty=uncertainty,selection=selection,profile=profile,
        model=model,backend=backend,dtype=dtype,device=device,priors=priors,sampling=sampling)
    folder.mkdir(parents=True,exist_ok=True)
    if profile.half_walkers <= len(p.parameter_names):raise ValueError('Full-rank half-cloud required')
    with _threads("numba",1,numba_threads):
        target=make_target(p,backend)
        dimension=len(p.parameter_names);default=initial_center(p)
        centers=np.tile(default,(p.cells,1)) if centers is None else np.array(centers,dtype=float,copy=True)
        if centers.shape!=(p.cells,dimension):raise ValueError('One correctly dimensioned center per cell required')
        from .numba_prototype import compile_specialization, _batch_structured
        before=time.perf_counter()
        dummy=torch.tensor(np.broadcast_to(centers,(profile.half_walkers,p.cells,dimension)).copy())
        compile_info=compile_specialization(dummy,*p.arguments,p.model,algorithm='structured',S=p.priors['S'],sampling=p.sampling)
        compile_info['compilation_or_cache_load_seconds']=time.perf_counter()-before
        before=time.perf_counter();target(dummy);one=make_target(_cell(p,[0]),backend);one(dummy[:1,:1].contiguous())
        disposable=initialize(target,(dummy,dummy.clone()),seed=seed+1,warmup=1)
        advance(target,disposable,2);_sync(p.y.device)
        compile_info['disposable_exact_shape_warmup_seconds']=time.perf_counter()-before
        signatures=list(map(str,_batch_structured.signatures));del dummy,disposable
        begin=time.perf_counter();optimizations=[]
        for g in range(p.cells):
            local=make_target(_cell(p,[g]),"numba")
            def objective(x):
                if time.time()-started>=wall_seconds:raise TimeoutError('Production initialization cap')
                return -float(local(torch.tensor(x.reshape(1,1,dimension))).sum())
            opt=minimize(objective,centers[g],method='Powell',options={'maxiter':60,'maxfev':2000,'xtol':1e-4,'ftol':1e-6})
            if not np.isfinite(opt.fun):raise ValueError('Finite high-density initialization required')
            centers[g]=opt.x;optimizations.append({'cell':g,'success':bool(opt.success),'evaluations':int(opt.nfev),'objective':float(opt.fun)})
            if (g+1)%12==0:print(f'optimized {g+1}/{p.cells} panels',flush=True)
        initialization=dict(method='per-panel Powell; independent full-rank 0.01 transformed-unit clouds',centers=centers.tolist(),optimizations=optimizations)
        _save_prepared(folder,p)
        config=dict(schema=1,backend=backend,runs=profile.runs,half_walkers=profile.half_walkers,warmup=profile.warmup,thin=profile.thin,
            seed=seed,chunk_sweeps=profile.chunk_sweeps,input_policy=validate_no_seasonal_input(data,model),source_identity=_source_identity(),production_source_identity=_identity(),
            torch=torch.__version__,runtime=_runtime(backend),threads={'torch':1,'numba':numba_threads},prepared_checksum=p.checksum,
            prepared_file_sha256=hashlib.sha256((folder/'prepared.npz').read_bytes()).hexdigest(),execution_profile=profile.to_dict(),selection=selection)
        config['provenance']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        for run in range(profile.runs):
            rng=np.random.default_rng(seed+run)
            halves=tuple(torch.tensor(centers+.01*rng.normal(size=(profile.half_walkers,p.cells,dimension))) for _ in range(2))
            for half in halves:
                for g in range(p.cells):
                    x=half[:,g].numpy()
                    if np.linalg.matrix_rank(x-x.mean(0))!=dimension:raise ValueError('Rank-deficient production cloud')
                if not bool(torch.isfinite(target(half)).all()):raise ValueError('Initial support failure')
            state=initialize(target,halves,seed=seed+10000+run,warmup=profile.warmup,thin=1,provenance=config['provenance'])
            save_state(folder/f'run-{run}'/'initial.npz',state)
        initialization['seconds']=time.perf_counter()-begin
        (folder/'initialization.json').write_text(json.dumps(initialization,indent=2)+'\n')
        (folder/'compilation.json').write_text(json.dumps(compile_info,indent=2)+'\n')
        fit=_continue_production(folder,p,config,target,profile.warmup+profile.retained_sweeps,started,wall_seconds)
        if list(map(str,_batch_structured.signatures))!=signatures:raise RuntimeError('Unaccounted compilation during production')
        return fit


def _continue_production(folder,p,config,target,end,started,wall_seconds):
    import fcntl
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        states=[]
        for run in range(config['runs']):
            paths=sorted((folder/f'run-{run}').glob('block-*.npz'))
            states.append(load_state(paths[-1] if paths else folder/f'run-{run}'/'initial.npz',provenance=config['provenance']))
        while min(s.iteration for s in states)<end and time.time()-started<wall_seconds:
            for run,state in enumerate(states):
                if state.iteration>=end or time.time()-started>=wall_seconds:continue
                count=min(config['chunk_sweeps'],end-state.iteration)
                before=time.perf_counter();next_state,draws,lp,_,steps=advance(target,state,count);elapsed=time.perf_counter()-before
                path=folder/f'run-{run}'/f'block-{state.iteration:09d}-{next_state.iteration:09d}.npz'
                if path.exists():raise ValueError('Never overwrite a sampler block')
                save_state(path,next_state,payload={'draws':draws.numpy(),'draw_logp':lp.numpy(),'retained_iterations':steps,'sampling_seconds':np.array(elapsed)})
                states[run]=next_state
                latest=dict(status='SAMPLING_COMPLETE_NOT_CONVERGENCE' if all(s.iteration>=end for s in states) else 'CHECKPOINTED',iterations=[s.iteration for s in states],target_iteration=end,actual_wall_seconds=time.time()-started)
                (folder/'latest.json').write_text(json.dumps(latest,indent=2)+'\n')
                print(f'production run{run}: {next_state.iteration}/{end}; block {elapsed:.3f}s',flush=True)
        if min(s.iteration for s in states)<end:
            latest['status']='WALL_CAP_CHECKPOINTED';(folder/'latest.json').write_text(json.dumps(latest,indent=2)+'\n')
    return FitResult(folder,p,config)


def resume_production(output_dir,*,retained_sweeps=None,wall_seconds=5400):
    fit=load_fit(output_dir);c=fit.config
    if c['backend'] in ('mps','cuda'):
        from .gpu_production import resume_gpu_production
        return resume_gpu_production(output_dir,retained_sweeps=retained_sweeps,wall_seconds=wall_seconds)
    if c.get('execution_profile',{}).get('name')!='production':raise ValueError('Production checkpoint required')
    if c['source_identity']!=_source_identity() or c['production_source_identity']!=_identity() or c['runtime']!=_runtime(c['backend']):
        raise ValueError('Exact production resume requires unchanged source/runtime')
    target_steps=c['execution_profile']['retained_sweeps'] if retained_sweeps is None else retained_sweeps
    if target_steps<c['execution_profile']['retained_sweeps']:raise ValueError('Never truncate saved evidence')
    with _threads(c['backend'],c['threads']['torch'],c['threads']['numba']):
        target=make_target(fit.prepared,c['backend']);state=load_state(fit.path/'run-0'/'initial.npz',provenance=c['provenance'])
        started=time.time();before=time.perf_counter();target(state.halves[0])
        (fit.path/f'resume-warmup-{time.time_ns()}.json').write_text(json.dumps({'compile_cache_and_disposable_warmup_seconds':time.perf_counter()-before})+'\n')
        return _continue_production(fit.path,fit.prepared,c,target,c['warmup']+target_steps,started,wall_seconds)


def production_parameter_cache(fit):
    """Create/reuse a memory-mapped hyperparameter window; no derived grid cube."""
    folder=fit.path;latest=json.loads((folder/'latest.json').read_text())
    if latest['status']!='SAMPLING_COMPLETE_NOT_CONVERGENCE':raise ValueError('Complete requested production window required')
    key=hashlib.sha256(json.dumps(latest['iterations']).encode()+fit.config['provenance'].encode()).hexdigest()[:16]
    path=folder/f'parameters-{key}.npy';r=fit.config['runs'];n=latest['target_iteration']-fit.config['warmup'];w=fit.config['half_walkers']*2
    if not path.exists():
        tmp=path.with_suffix('.tmp.npy');values=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float64,shape=(r,n,w,fit.prepared.cells,len(fit.prepared.parameter_names)))
        for run in range(r):
            offset=0
            for file in sorted((folder/f'run-{run}').glob('block-*.npz')):
                with np.load(file) as z:
                    v=z['draws'];values[run,offset:offset+len(v)]=v;offset+=len(v)
            if offset!=n:raise ValueError('Incomplete retained window')
        values.flush();del values;tmp.replace(path)
    return np.load(path,mmap_mode='r')


def diagnose_production(fit):
    """Visible per-cell warnings, not Stan equivalence or convergence acceptance."""
    if fit.config.get('execution_profile',{}).get('name')!='production':raise ValueError('Production fit required')
    u=production_parameter_cache(fit);rows=[]
    states=[load_state(sorted((fit.path/f'run-{r}').glob('block-*.npz'))[-1],provenance=fit.config['provenance']) for r in range(fit.config['runs'])]
    for g in range(fit.prepared.cells):
        names,x=physical_parameters(np.asarray(u[:,:,:,g]),fit.prepared.model,fit.prepared.regressors.shape[-1]);names,x=finite_moment_summaries(names,x)
        r,t,w,p=x.shape;avg=x.mean(2)
        tau=np.array([[autocorrelation_time(avg[i,:,j]) for j in range(p)] for i in range(r)])
        mcvar=(avg.var(1,ddof=1)*tau/t).sum(0)/r**2;ess=np.minimum(r*t*w,x.var((0,1,2),ddof=1)/mcvar)
        first=x[:,:t//2];ratio=x.std((0,1,2),ddof=1)/first.std((0,1,2),ddof=1)
        delta=(x.mean((0,1,2))-first.mean((0,1,2)))/x.std((0,1,2),ddof=1)
        acceptance=np.stack([(s.accepted[:,:,g].sum(0)/s.proposed[:,:,g].sum(0)).cpu().numpy() for s in states])
        flags=[]
        if r==1:flags.append('SINGLE_ENSEMBLE_NO_BETWEEN_RUN_RHAT')
        if np.any(t<50*tau):flags.append('SHORT_AUTOCORRELATION_ESTIMATE')
        if np.any(ess<2000):flags.append('NOMINAL_MEAN_ESS_BELOW_2000')
        if np.any(abs(ratio-1)>.05) or np.any(abs(delta)>.05):flags.append('HALF_FULL_STABILITY_POINT_FLAG')
        if np.any(acceptance==0):flags.append('ZERO_ACCEPTANCE_WALKER')
        rows.append(dict(cell=g,panel=str(fit.prepared.cell_coordinates['panel'][g]),flags=flags,mean_ess_nominal=ess.tolist(),maximum_iat=float(tau.max()),full_to_half_sd_ratio=ratio.tolist(),mean_change_in_sd=delta.tolist(),acceptance_min=float(acceptance.min()),acceptance_mean=float(acceptance.mean()),between_run_rhat=None))
    return dict(status='PRODUCTION_ESTIMATES_WITH_FLAGS',qualification_eligible=False,quantities=names,cells=rows,
        scope='Means/widths/acceptance monitoring only; no tail ESS certificate, four-run qualification or all-pairs materialization')
