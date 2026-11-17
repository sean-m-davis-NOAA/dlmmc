"""Production GPU32 sampling with explicit hardware and scientific status.

User-approved production availability does not erase numerical or single-ensemble
qualification flags. CUDA is implemented but native validation remains pending.
"""
from dataclasses import replace
from pathlib import Path
from .input_contract import validate_no_seasonal_input
import hashlib, json, time
import numpy as np
import torch
from scipy.optimize import minimize
from affine.torch_stream import initialize, advance, save_state, load_state
from .structured_mps import StructuredMPSTarget, compile_kernel
from .profiles import PRODUCTION_PROFILE
from .production import _cell, prepare_production, initial_center
from .profiles import validate_production_scope
from .compute import resolve_compute, gpu_arithmetic
from .sampling import _sync
from .sampling import (FitResult, make_target, _save_prepared, _load_prepared,
    _source_identity, _runtime, _threads, load_fit)


def identity():
    return {**_source_identity(), **{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
        for n in ('structured_mps.py','gpu_production.py','production.py','compute.py','profiles.py')}}


def gpu_prepared(p,device,dtype=torch.float32):
    """Explicit round/copy of canonical prepared inputs; no normalization change."""
    return replace(p, **{n:getattr(p,n).to(device=device,dtype=torch.bool if n=='observed' else dtype).contiguous()
        for n in ('y','variance','observed','regressors')})


def gpu_target(p,backend):
    if backend=='mps':return StructuredMPSTarget(p)
    from .torch_filter import TorchTarget
    return TorchTarget(p)


def continue_blocks(folder, target, state, *, deadline, end, chunk_sweeps=16):
    """Full-batch MPS transitions; synchronize before timing and host checkpoint."""
    folder=Path(folder)
    while state.iteration<end:
        if time.monotonic()>=deadline:raise TimeoutError('GPU fit cap; complete checkpoints retained')
        count=min(chunk_sweeps,end-state.iteration)
        _sync(state.halves[0].device);before=time.perf_counter()
        new,draws,lp,_,steps=advance(target,state,count)
        _sync(state.halves[0].device);elapsed=time.perf_counter()-before
        path=folder/'run-0'/f'block-{state.iteration:09d}-{new.iteration:09d}.npz'
        if path.exists():raise ValueError('Never overwrite GPU checkpoint')
        io=time.perf_counter()
        save_state(path,new,payload={'draws':draws.cpu().numpy(),'draw_logp':lp.cpu().numpy(),
            'retained_iterations':steps,'sampling_seconds':np.array(elapsed)})
        checkpoint_seconds=time.perf_counter()-io
        latest=dict(status='SAMPLING_COMPLETE_NOT_CONVERGENCE' if new.iteration==end else 'CHECKPOINTED',
            iterations=[new.iteration],target_iteration=end,last_checkpoint_seconds=checkpoint_seconds)
        (folder/'latest.json').write_text(json.dumps(latest,indent=2)+'\n')
        print(f'GPU full batch: {new.iteration}/{end}; {elapsed:.3f}s; save {checkpoint_seconds:.3f}s',flush=True)
        state=new
    return state


@torch.no_grad()
def fit_gpu_production(data, regressors, *, uncertainty, output_dir, selection=None,
        seed=2026111801,wall_seconds=5400,profile=PRODUCTION_PROFILE,centers=None,
        model='vanilla_ar1',backend='mps',device=None,dtype=None,priors=None,sampling='monthly',numba_threads=8):
    started=time.monotonic();deadline=started+wall_seconds
    policy=resolve_compute(backend,device=device,dtype=dtype)
    validate_production_scope(data.panel.values,selection,model=model,backend=backend,dtype=policy.dtype,profile=profile)
    if profile.runs!=1:raise ValueError('GPU production currently supports one ensemble; qualification remains separate')
    # Scope was checked in GPU coordinates above; this translated copy is only
    # for canonical CPU64 input preparation and optimized initialization.
    canonical=None if selection is None else dict(selection,backend='numba',dtype='float64')
    p=prepare_production(data,regressors,uncertainty=uncertainty,selection=canonical,profile=profile,
        model=model,priors=priors,sampling=sampling)
    from .prior_specification import announce_model
    announce_model(p.model,p.regressor_names,p.priors,panels=p.cells)
    dimension=len(p.parameter_names)
    if profile.half_walkers<=dimension:raise ValueError('Full-rank half-cloud required')
    folder=Path(output_dir).resolve()
    if folder.exists():raise ValueError('Preserve prior fit; use resume or a new path')
    folder.mkdir(parents=True);oracle=folder/'oracle-input';oracle.mkdir();_save_prepared(oracle,p)
    with _threads('numba',1,numba_threads), gpu_arithmetic(policy.device):
        before=time.perf_counter();pg=gpu_prepared(p,policy.device,getattr(torch,policy.dtype));target=gpu_target(pg,backend);_sync(pg.y.device)
        preparation=time.perf_counter()-before
        from .numba_prototype import compile_specialization, _batch_structured
        dummy=torch.tensor(np.broadcast_to(initial_center(p),(profile.half_walkers,p.cells,dimension)).copy())
        before=time.perf_counter();metal_seconds=compile_kernel(p.model,p.regressors.shape[-1])[1] if backend=='mps' else 0.
        cpu_info=compile_specialization(dummy,*p.arguments,p.model,algorithm='structured',S=p.priors['S'],sampling=p.sampling)
        compilation=time.perf_counter()-before
        before=time.perf_counter();one=make_target(_cell(p,[0]),'numba');one(dummy[:1,:1].contiguous())
        dg=dummy.to(policy.device,pg.y.dtype)
        target(dg); disposable=initialize(target,(dg,dg.clone()),seed=seed+1,warmup=1)
        advance(target,disposable,2);_sync(pg.y.device);warm=time.perf_counter()-before
        signatures=list(map(str,_batch_structured.signatures));del disposable,dummy,dg
        begin=time.perf_counter();centers=np.tile(initial_center(p),(p.cells,1)) if centers is None else np.asarray(centers,dtype=float).copy();optimizations=[]
        if centers.shape!=(p.cells,dimension):raise ValueError('One correctly dimensioned center per panel required')
        for g in range(p.cells):
            local=make_target(_cell(p,[g]),'numba')
            def objective(x):
                if time.monotonic()>=deadline:raise TimeoutError('GPU fit initialization cap')
                return -float(local(torch.tensor(x.reshape(1,1,dimension))).sum())
            opt=minimize(objective,centers[g],method='Powell',options={'maxiter':60,'maxfev':2000,'xtol':1e-4,'ftol':1e-6})
            if not np.isfinite(opt.fun):raise ValueError('Nonfinite optimized center')
            centers[g]=opt.x;optimizations.append(dict(cell=g,success=bool(opt.success),evaluations=int(opt.nfev),objective=float(opt.fun)))
            if (g+1)%24==0:print(f'CPU64 initialization: {g+1}/{p.cells}',flush=True)
        _save_prepared(folder,pg)
        config=dict(schema=1,backend=backend,runs=1,half_walkers=profile.half_walkers,warmup=profile.warmup,thin=1,seed=seed,chunk_sweeps=profile.chunk_sweeps,
            source_identity=identity(),runtime=_runtime('numba'),threads={'torch':1,'numba':numba_threads},
            prepared_checksum=p.checksum,prepared_file_sha256=hashlib.sha256((folder/'prepared.npz').read_bytes()).hexdigest(),
            oracle_input_sha256=hashlib.sha256((oracle/'prepared.npz').read_bytes()).hexdigest(),
            execution_profile=profile.to_dict(),selection=selection,qualification_eligible=False,
            seasonal_input_policy=validate_no_seasonal_input(data,model),
            sampling_dtype=policy.dtype,sampling_device=policy.device,reconstruction_dtype=policy.dtype,reconstruction_device=policy.device,compute_validation=policy.validation,
            precision_gate=('NUMERICAL_FAIL_RETAINED; user-approved availability is not qualification' if policy.dtype=='float32' else 'CUDA64_NATIVE_VALIDATION_PENDING'),
            input_policy='Normalize in canonical float64, explicitly cast GPU inputs; native-dtype GPU reconstruction by default; canonical CPU64 inputs retained for explicit oracle comparisons')
        config['provenance']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        rng=np.random.default_rng(seed)
        halves=tuple(torch.tensor(centers+.01*rng.normal(size=(profile.half_walkers,p.cells,dimension)),dtype=pg.y.dtype,device=policy.device) for _ in range(2))
        for half in halves:
            actual=half.cpu().numpy()
            for g in range(p.cells):
                if np.linalg.matrix_rank(actual[:,g]-actual[:,g].mean(0))!=dimension:raise ValueError('Actual sampling-dtype cloud rank failure')
        state=initialize(target,halves,seed=seed+10000,warmup=profile.warmup,thin=1,provenance=config['provenance'])
        save_state(folder/'run-0'/'initial.npz',state);_sync(state.halves[0].device)
        (folder/'initialization.json').write_text(json.dumps(dict(seconds=time.perf_counter()-begin,
            method='Fresh CPU64 Powell centers, then actual-GPU-dtype full-rank/support checks, 0.01 cloud',
            centers=centers.tolist(),optimizations=optimizations),indent=2)+'\n')
        (folder/'compilation.json').write_text(json.dumps(dict(compilation_or_cache_load_seconds=compilation,
            disposable_exact_shape_warmup_seconds=warm,metal_compile_pipeline_seconds=metal_seconds,
            cpu_initializer_compile=cpu_info,device_preparation_seconds=preparation),indent=2)+'\n')
        continue_blocks(folder,target,state,deadline=deadline,end=profile.warmup+profile.retained_sweeps,chunk_sweeps=profile.chunk_sweeps)
        if signatures!=list(map(str,_batch_structured.signatures)):raise RuntimeError('Unexpected initializer compilation')
    return FitResult(folder,pg,config)


@torch.no_grad()
def resume_gpu_production(output_dir, *, retained_sweeps=None, wall_seconds=5400):
    fit=load_fit(output_dir);c=fit.config
    if c['backend'] not in ('mps','cuda') or c['source_identity']!=identity() or c['runtime']!=_runtime('numba'):
        raise ValueError('Exact GPU resume requires unchanged source, runtime and backend')
    started=time.monotonic();target=gpu_target(fit.prepared,c['backend'])
    paths=sorted((fit.path/'run-0').glob('block-*.npz'))
    state=load_state(paths[-1] if paths else fit.path/'run-0/initial.npz',provenance=c['provenance'])
    before=time.perf_counter();target(state.halves[0]);_sync(state.halves[0].device)
    (fit.path/f'resume-warmup-{time.time_ns()}.json').write_text(json.dumps(dict(
        compile_cache_and_disposable_warmup_seconds=time.perf_counter()-before))+'\n')
    end=c['execution_profile']['retained_sweeps'] if retained_sweeps is None else retained_sweeps
    if end<c['execution_profile']['retained_sweeps']:raise ValueError('Never truncate saved evidence')
    with _threads('numba',c['threads']['torch'],c['threads']['numba']), gpu_arithmetic(fit.prepared.y.device):
        continue_blocks(fit.path,target,state,deadline=started+wall_seconds,end=c['warmup']+end,chunk_sweeps=c['chunk_sweeps'])
    return fit


def cpu64_reconstruction_view(fit):
    """Explicit analysis-only view, never used for GPU sampling/continuation."""
    path=fit.path/'oracle-input'
    if hashlib.sha256((path/'prepared.npz').read_bytes()).hexdigest()!=fit.config['oracle_input_sha256']:
        raise ValueError('Canonical reconstruction inputs changed')
    p=_load_prepared(path)
    if p.checksum!=fit.prepared.checksum:raise ValueError('Input identity mismatch')
    return replace(fit,prepared=p)
