"""Explicit experimental GPU fit with a separately declared CPU64 reconstruction.

This does not promote float32 to a qualified production backend. The numerical
failures and single-ensemble limitations remain visible in the saved config.
"""
from dataclasses import replace
from pathlib import Path
import hashlib, json, time
import numpy as np
import torch
from scipy.optimize import minimize
from affine.torch_stream import initialize, advance, save_state, load_state
from .structured_mps import StructuredMPSTarget, compile_kernel
from .profiles import LEGACY_PRODUCTION_PROFILE as PRODUCTION_PROFILE
from .production import _cell
from .sampling import (FitResult, make_target, _save_prepared, _load_prepared,
    _source_identity, _runtime, _threads, load_fit)


def identity():
    return {**_source_identity(), **{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
        for n in ('structured_mps.py','structured_mps_sampling.py','production.py')}}


def gpu_prepared(p):
    """Explicit round/copy of canonical prepared inputs; no normalization change."""
    return replace(p, **{n:getattr(p,n).to(device='mps',dtype=torch.bool if n=='observed' else torch.float32).contiguous()
        for n in ('y','variance','observed','regressors')})


def validate_scope(p, selection, profile):
    if p.model!='vanilla_ar1' or p.y.device.type!='cpu' or p.y.dtype!=torch.float64 or p.regressors.shape[-1]!=5:
        raise ValueError('Canonical CPU64 vanilla AR1 / five-regressor inputs required')
    if profile!=PRODUCTION_PROFILE:
        raise ValueError('This named research fit retains the exact 128/128/512 production settings')
    if (not isinstance(selection,dict) or selection.get('scope')!='structured_mps_research_fit'
        or not selection.get('explicit_user_response') or selection.get('backend')!='structured_mps'
        or selection.get('dtype')!='float32' or selection.get('profile')!=profile.to_dict()
        or selection.get('panel_ids')!=list(map(str,p.cell_coordinates['panel']))):
        raise ValueError('Matching explicit experimental GPU panel selection required')


def continue_blocks(folder, target, state, *, deadline, end=640):
    """Full-batch MPS transitions; synchronize before timing and host checkpoint."""
    folder=Path(folder)
    while state.iteration<end:
        if time.monotonic()>=deadline:raise TimeoutError('GPU fit cap; complete checkpoints retained')
        count=min(16,end-state.iteration)
        torch.mps.synchronize();before=time.perf_counter()
        new,draws,lp,_,steps=advance(target,state,count)
        torch.mps.synchronize();elapsed=time.perf_counter()-before
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
def fit_structured_mps_research(prepared64, *, output_dir, selection, seed=2026111801,
        wall_seconds=5400, profile=PRODUCTION_PROFILE):
    started=time.monotonic();deadline=started+wall_seconds;p=prepared64
    validate_scope(p,selection,profile)
    folder=Path(output_dir).resolve()
    if folder.exists():raise ValueError('Preserve prior fit; use resume or a new path')
    folder.mkdir(parents=True);oracle=folder/'oracle-input';oracle.mkdir();_save_prepared(oracle,p)
    with _threads('numba',1,8):
        before=time.perf_counter();pg=gpu_prepared(p);target=StructuredMPSTarget(pg);torch.mps.synchronize()
        preparation=time.perf_counter()-before
        from .numba_prototype import compile_specialization, _batch_structured
        dummy=torch.tensor(np.broadcast_to([-9.2,-4.6,-2.3,0.],(64,p.cells,4)).copy())
        before=time.perf_counter();_,metal_seconds=compile_kernel()
        cpu_info=compile_specialization(dummy,*p.arguments,p.model,algorithm='structured',S=p.priors['S'],sampling=p.sampling)
        compilation=time.perf_counter()-before
        before=time.perf_counter();one=make_target(_cell(p,[0]),'numba');one(dummy[:1,:1].contiguous())
        dg=dummy.to('mps',torch.float32)
        target(dg); disposable=initialize(target,(dg,dg.clone()),seed=seed+1,warmup=1)
        advance(target,disposable,2);torch.mps.synchronize();warm=time.perf_counter()-before
        signatures=list(map(str,_batch_structured.signatures));del disposable,dummy,dg
        begin=time.perf_counter();centers=np.tile([-9.2,-4.6,-2.3,0.],(p.cells,1));optimizations=[]
        for g in range(p.cells):
            local=make_target(_cell(p,[g]),'numba')
            def objective(x):
                if time.monotonic()>=deadline:raise TimeoutError('GPU fit initialization cap')
                return -float(local(torch.tensor(x.reshape(1,1,4))).sum())
            opt=minimize(objective,centers[g],method='Powell',options={'maxiter':60,'maxfev':2000,'xtol':1e-4,'ftol':1e-6})
            if not np.isfinite(opt.fun):raise ValueError('Nonfinite optimized center')
            centers[g]=opt.x;optimizations.append(dict(cell=g,success=bool(opt.success),evaluations=int(opt.nfev),objective=float(opt.fun)))
            if (g+1)%24==0:print(f'CPU64 initialization: {g+1}/{p.cells}',flush=True)
        _save_prepared(folder,pg)
        config=dict(schema=1,backend='structured_mps',runs=1,half_walkers=64,warmup=128,thin=1,seed=seed,chunk_sweeps=16,
            source_identity=identity(),runtime=_runtime('numba'),threads={'torch':1,'numba':8},
            prepared_checksum=p.checksum,prepared_file_sha256=hashlib.sha256((folder/'prepared.npz').read_bytes()).hexdigest(),
            oracle_input_sha256=hashlib.sha256((oracle/'prepared.npz').read_bytes()).hexdigest(),
            execution_profile=profile.to_dict(),selection=selection,qualification_eligible=False,
            sampling_dtype='float32',sampling_device='mps',reconstruction_dtype='float64',reconstruction_device='cpu',
            precision_gate='FAIL_RETAINED_RESEARCH_ONLY; reports/14-structured-mps-results.json',
            input_policy='Normalize in canonical float64, explicitly cast GPU inputs; reconstruct with original CPU64 inputs')
        config['provenance']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        rng=np.random.default_rng(seed)
        halves=tuple(torch.tensor(centers+.01*rng.normal(size=(64,p.cells,4)),dtype=torch.float32,device='mps') for _ in range(2))
        for half in halves:
            actual=half.cpu().numpy()
            for g in range(p.cells):
                if np.linalg.matrix_rank(actual[:,g]-actual[:,g].mean(0))!=4:raise ValueError('Actual float32 cloud rank failure')
        state=initialize(target,halves,seed=seed+10000,warmup=128,thin=1,provenance=config['provenance'])
        save_state(folder/'run-0'/'initial.npz',state);torch.mps.synchronize()
        (folder/'initialization.json').write_text(json.dumps(dict(seconds=time.perf_counter()-begin,
            method='Fresh CPU64 Powell centers, then actual-MPS32 full-rank/support checks, 0.01 cloud',
            centers=centers.tolist(),optimizations=optimizations),indent=2)+'\n')
        (folder/'compilation.json').write_text(json.dumps(dict(compilation_or_cache_load_seconds=compilation,
            disposable_exact_shape_warmup_seconds=warm,metal_compile_pipeline_seconds=metal_seconds,
            cpu_initializer_compile=cpu_info,device_preparation_seconds=preparation),indent=2)+'\n')
        continue_blocks(folder,target,state,deadline=deadline)
        if signatures!=list(map(str,_batch_structured.signatures)):raise RuntimeError('Unexpected initializer compilation')
    return FitResult(folder,pg,config)


@torch.no_grad()
def resume_structured_mps_research(output_dir, *, wall_seconds):
    fit=load_fit(output_dir);c=fit.config
    if c['backend']!='structured_mps' or c['source_identity']!=identity() or c['runtime']!=_runtime('numba'):
        raise ValueError('Exact GPU resume requires unchanged source, runtime and backend')
    started=time.monotonic();target=StructuredMPSTarget(fit.prepared)
    paths=sorted((fit.path/'run-0').glob('block-*.npz'))
    state=load_state(paths[-1] if paths else fit.path/'run-0/initial.npz',provenance=c['provenance'])
    before=time.perf_counter();target(state.halves[0]);torch.mps.synchronize()
    (fit.path/f'resume-warmup-{time.time_ns()}.json').write_text(json.dumps(dict(
        compile_cache_and_disposable_warmup_seconds=time.perf_counter()-before))+'\n')
    continue_blocks(fit.path,target,state,deadline=started+wall_seconds)
    return fit


def cpu64_reconstruction_view(fit):
    """Explicit analysis-only view, never used for GPU sampling/continuation."""
    path=fit.path/'oracle-input'
    if hashlib.sha256((path/'prepared.npz').read_bytes()).hexdigest()!=fit.config['oracle_input_sha256']:
        raise ValueError('Canonical reconstruction inputs changed')
    p=_load_prepared(path)
    if p.checksum!=fit.prepared.checksum:raise ValueError('Input identity mismatch')
    return replace(fit,prepared=p)
