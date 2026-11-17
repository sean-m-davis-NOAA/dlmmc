"""Public affine fitting with bounded device storage and exact local resume."""
from .components import initial_coordinates
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import fcntl
from contextlib import contextmanager
import importlib.metadata
import platform
import numpy as np
import torch
import xarray as xr
from scipy.optimize import minimize
from affine.torch_stream import initialize,advance,save_state,load_state
from .prepare import PreparedData,prepare_data
from .torch_filter import TorchTarget


def _jsonable(x):
    if isinstance(x,dict):return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_jsonable(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x


def _sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)
    elif device.type=='mps':torch.mps.synchronize()


class NumbaTarget:
    def __init__(self,p):
        if p.y.device.type!='cpu' or p.y.dtype!=torch.float64:raise ValueError('Numba backend requires explicit CPU float64')
        self.prepared=p;self.correction=TorchTarget(p)

    def __call__(self,u):
        from .numba_prototype import target_numba
        p=self.prepared
        value=target_numba(u,*p.arguments,p.model,S=p.priors['S'],sampling=p.sampling,algorithm='structured')
        return value+self.correction.prior_correction(u,torch.isfinite(value))


def make_target(prepared,backend):
    if backend=='mps':
        from .structured_mps import StructuredMPSTarget
        return StructuredMPSTarget(prepared)
    if backend=='cuda':
        if prepared.y.device.type!='cuda' or prepared.y.dtype!=torch.float32:raise ValueError('CUDA backend requires native float32')
        return TorchTarget(prepared)
    if backend=='torch':return TorchTarget(prepared)
    if backend=='numba':return NumbaTarget(prepared)
    raise ValueError('Choose numba, torch, mps or cuda backend')


def optimized_clouds(prepared,target,*,half_walkers=64,runs=4,seed=20261104,centers=None,spread=.01,deadline=None):
    """Optimize transformed coordinates, then check rank/support in actual dtype."""
    p=prepared;dimension=len(p.parameter_names)
    if half_walkers<dimension+1 or runs<4 or spread<=0:raise ValueError('Full-rank halves and four independent runs required')
    default=initial_coordinates(p.model,p.regressors.shape[-1])
    centers=np.broadcast_to(default,(p.cells,dimension)).copy() if centers is None else np.asarray(centers,dtype=float).copy()
    if centers.shape!=(p.cells,dimension):raise ValueError('One transformed center per cell required')
    optimization=[]
    # Optimize all independent cells jointly; likelihood remains factorized.
    # Powell uses function values rather than unreliable float32 finite differences.
    def objective(flat):
        if deadline is not None and time.time()>=deadline:
            raise TimeoutError('Initialization reached the actual wall-time cap; no sampling launched')
        with torch.no_grad():value=target(torch.as_tensor(flat.reshape(1,p.cells,dimension),dtype=p.y.dtype,device=p.y.device))
        return -float(value.sum().cpu())
    result=minimize(objective,centers.ravel(),method='Powell',options={'maxiter':60,'maxfev':2000,'xtol':1e-4,'ftol':1e-6})
    if not np.isfinite(result.fun):raise ValueError('No finite optimized initial point')
    centers=result.x.reshape(p.cells,dimension)
    optimization.append(dict(success=bool(result.success),message=str(result.message),evaluations=int(result.nfev),objective=float(result.fun)))
    halves=[]
    for run in range(runs):
        rng=np.random.default_rng(seed+run)
        pair=[]
        for _ in range(2):
            x=torch.as_tensor(centers+spread*rng.normal(size=(half_walkers,p.cells,dimension)),dtype=p.y.dtype,device=p.y.device)
            actual=x.cpu().numpy()
            for cell in range(p.cells):
                if np.linalg.matrix_rank(actual[:,cell]-actual[:,cell].mean(0))<dimension:raise ValueError('Initial half-cloud is rank deficient in actual dtype')
            if not bool(torch.isfinite(target(x)).all()):raise ValueError('Initial cloud leaves finite support; no draw rejection')
            pair.append(x)
        halves.append(tuple(pair))
    return halves,dict(method='tight full-rank cloud near optimized transformed point',spread=spread,centers=centers.tolist(),optimization=optimization)


def _save_prepared(folder,p):
    metadata=dict(model=p.model,offset=p.offset.tolist(),scale=p.scale.tolist(),regressor_names=p.regressor_names,
                  priors=_jsonable(p.priors),sampling=p.sampling,checksum=p.checksum,input_policy=p.input_policy,
                  cell_coordinate_keys=list(p.cell_coordinates),device=str(p.y.device),dtype=str(p.y.dtype).split('.')[-1])
    arrays={name:val.cpu().numpy() for name,val in zip(('y','variance','observed','regressors'),p.arguments)}
    arrays['time']=p.time
    for j,val in enumerate(p.cell_coordinates.values()):
        arrays[f'coord{j}']=val.astype(str) if val.dtype.kind=='O' else val
    with (folder/'prepared.tmp').open('wb') as f:np.savez(f,metadata=json.dumps(metadata),**arrays)
    (folder/'prepared.tmp').replace(folder/'prepared.npz')


def _load_prepared(folder):
    with np.load(folder/'prepared.npz',allow_pickle=False) as z:
        m=json.loads(str(z['metadata']));dtype=getattr(torch,m['dtype']);device=torch.device(m['device'])
        tensors=[torch.as_tensor(z[n].copy(),dtype=torch.bool if n=='observed' else dtype,device=device)
                 for n in ('y','variance','observed','regressors')]
        coords={name:z[f'coord{j}'].copy() for j,name in enumerate(m['cell_coordinate_keys'])}
        return PreparedData(m['model'],*tensors,np.asarray(m['offset']),np.asarray(m['scale']),z['time'].copy(),coords,
                            m['regressor_names'],m['priors'],m['sampling'],m['checksum'],m.get('input_policy'))


def _source_identity():
    import affine.torch_stream
    paths=[Path(__file__),Path(affine.torch_stream.__file__)]
    paths.extend(Path(__file__).parent/name for name in ('prepare.py','torch_filter.py','tensor_target.py','numba_prototype.py','transforms.py','api.py','input_contract.py','preprocessing.py','__init__.py','compute.py','structured_mps.py','components.py'))
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


@dataclass
class FitResult:
    path: Path
    prepared: PreparedData
    config: dict

    def posterior(self):
        """Materialize parameter draws on CPU; filtering histories are never stored."""
        runs=[];logs=[];iterations=None
        for run in range(self.config['runs']):
            parts=[];lp=[];it=[]
            for path in sorted((self.path/f'run-{run}').glob('block-*.npz')):
                with np.load(path,allow_pickle=False) as z:
                    parts.append(z['draws']);lp.append(z['draw_logp']);it.append(z['retained_iterations'])
            runs.append(np.concatenate(parts));logs.append(np.concatenate(lp));current=np.concatenate(it)
            if iterations is not None and not np.array_equal(iterations,current):raise ValueError('Runs have different retained windows; resume incomplete fit')
            iterations=current
        values=np.stack(runs);logp=np.stack(logs);p=self.prepared
        coords={'run':np.arange(len(runs)),'draw':iterations,'walker':np.arange(values.shape[2]),
                'cell':np.arange(p.cells),'parameter':p.parameter_names}
        for name,array in p.cell_coordinates.items():coords[name]=('cell',array)
        return xr.Dataset({'unconstrained':(('run','draw','walker','cell','parameter'),values),
                           'log_probability':(('run','draw','walker','cell'),logp)},coords=coords,
            attrs={'model':p.model,'dtype':str(p.y.dtype),'device':str(p.y.device),'backend':self.config['backend'],
                   'policy':'legacy fixed variances; explicit mask; original priors with explicit configured scales',
                   'diagnostic_status':'NOT_YET_DIAGNOSED','checkpoint_directory':str(self.path),
                   'walkers_are_independent_chains':'false','warmup_transitions':self.config['warmup']})


def _sample_parameters(prepared,*,output_dir,backend='numba',retained_sweeps=1000,warmup=100,
                      half_walkers=64,runs=4,thin=1,chunk_sweeps=50,seed=20261104,
                      wall_seconds=900,centers=None,initial_halves=None):
    """Start a bounded fit. Existing directories must use resume_fit explicitly.

    Precompile/prewarm the exact callback separately. Operational wall limits
    include that time; measured sampling excludes it and includes MCMC warmup.
    Initial halves are an explicit advanced/testing option, checked by adapter.
    """
    started=time.time();folder=Path(output_dir).resolve()
    if (folder/'config.json').exists():raise ValueError('Existing fit: use resume_fit')
    if min(retained_sweeps,chunk_sweeps)<1 or runs<4 or thin<1 or wall_seconds<=0:raise ValueError('Invalid sampling budget')
    folder.mkdir(parents=True,exist_ok=True)
    from .compute import prepared_for_backend
    prepared=prepared_for_backend(prepared,backend)
    target=make_target(prepared,backend)
    dummy=prepared.y.new_tensor(initial_coordinates(prepared.model,prepared.regressors.shape[-1])).expand(half_walkers,prepared.cells,-1).clone()
    compile_info={'compilation_or_cache_load_seconds':0.}
    if backend=='numba':
        from .numba_prototype import compile_specialization
        compile_info=compile_specialization(dummy,*prepared.arguments,prepared.model,algorithm='structured',S=prepared.priors['S'],sampling=prepared.sampling)
    elif backend=='mps':
        from .structured_mps import compile_kernel
        compile_info['compilation_or_cache_load_seconds']=compile_kernel(prepared.model,prepared.regressors.shape[-1])[1]
    _sync(prepared.y.device);before=time.perf_counter();target(dummy);_sync(prepared.y.device)
    compile_info['disposable_exact_shape_warmup_seconds']=time.perf_counter()-before
    init_start=time.perf_counter()
    if initial_halves is None:
        try:
            halves,initialization=optimized_clouds(prepared,target,half_walkers=half_walkers,runs=runs,seed=seed,centers=centers,deadline=started+wall_seconds)
        except TimeoutError:
            (folder/'initialization-timeout.json').write_text(json.dumps({'status':'ACTUAL_WALL_CAP_DURING_INITIALIZATION','actual_wall_seconds':time.time()-started,'compilation':compile_info})+'\n')
            raise
    else:
        if len(initial_halves)!=runs:raise ValueError('Initial half-pairs must match independent runs')
        for pair in initial_halves:
            if len(pair)!=2:raise ValueError('Exactly two halves required')
            for half in pair:
                if tuple(half.shape)!=(half_walkers,prepared.cells,len(prepared.parameter_names)):
                    raise ValueError('Initial halves do not match declared shape')
                for cell in range(prepared.cells):
                    z=half[:,cell].cpu().numpy()
                    if np.linalg.matrix_rank(z-z.mean(0))<len(prepared.parameter_names):
                        raise ValueError('Caller-supplied initial half is rank deficient')
        halves=initial_halves;initialization={'method':'explicit caller-supplied halves; no automatic optimization'}
    initialization['seconds']=time.perf_counter()-init_start
    _save_prepared(folder,prepared)
    config=dict(schema=1,backend=backend,runs=runs,half_walkers=half_walkers,warmup=warmup,thin=thin,
                seed=seed,chunk_sweeps=chunk_sweeps,source_identity=_source_identity(),torch=torch.__version__,
                runtime=_runtime(backend),threads={'torch':torch.get_num_threads(),'numba':__import__('numba').get_num_threads() if backend=='numba' else None},
                prepared_checksum=prepared.checksum,prepared_file_sha256=hashlib.sha256((folder/'prepared.npz').read_bytes()).hexdigest())
    config['provenance']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (folder/'compilation.json').write_text(json.dumps(compile_info,indent=2)+'\n')
    (folder/'initialization.json').write_text(json.dumps(initialization,indent=2)+'\n')
    for run in range(runs):
        state=initialize(target,halves[run],seed=seed+10000+run,warmup=warmup,thin=thin,provenance=config['provenance'])
        save_state(folder/f'run-{run}'/'initial.npz',state)
    return _continue(folder,prepared,config,target,warmup+retained_sweeps,started,wall_seconds)


def _continue(folder,p,config,target,end,started,wall_seconds):
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        states=[]
        for run in range(config['runs']):
            blocks=sorted((folder/f'run-{run}').glob('block-*.npz'))
            states.append(load_state(blocks[-1] if blocks else folder/f'run-{run}'/'initial.npz',provenance=config['provenance']))
        # Round-robin chunks keep independent runs at comparable lengths at cap.
        while min(s.iteration for s in states)<end and time.time()-started<wall_seconds:
            for run,state in enumerate(states):
                if state.iteration>=end:continue
                if time.time()-started>=wall_seconds:break
                count=min(config['chunk_sweeps'],end-state.iteration)
                _sync(p.y.device);before=time.perf_counter()
                next_state,draws,lp,extras,steps=advance(target,state,count)
                _sync(p.y.device);elapsed=time.perf_counter()-before
                path=folder/f'run-{run}'/f'block-{state.iteration:09d}-{next_state.iteration:09d}.npz'
                if path.exists():raise ValueError('Existing block would be overwritten')
                save_state(path,next_state,payload={'draws':draws.cpu().numpy(),'draw_logp':lp.cpu().numpy(),
                    'retained_iterations':steps,'sampling_seconds':np.array(elapsed)})
                states[run]=next_state
        status=dict(status='SAMPLING_COMPLETE_NOT_CONVERGENCE' if all(s.iteration>=end for s in states) else 'WALL_CAP_CHECKPOINTED',
                    iterations=[s.iteration for s in states],target_iteration=end,actual_wall_seconds=time.time()-started)
        (folder/'latest.json').write_text(json.dumps(status,indent=2)+'\n')
    return FitResult(folder,p,config)


def _resume_fit(output_dir,*,retained_sweeps,wall_seconds=900):
    """Continue to an absolute retained-sweep target with identical saved policy."""
    started=time.time();folder=Path(output_dir).resolve();config=json.loads((folder/'config.json').read_text())
    _verify_config(config)
    if config['source_identity']!=_source_identity() or config['torch']!=torch.__version__:
        raise ValueError('Source/runtime changed; exact continuation needs explicit migration review')
    if config['runtime']!=_runtime(config['backend']):raise ValueError('Runtime/hardware changed; exact continuation not certified')
    if hashlib.sha256((folder/'prepared.npz').read_bytes()).hexdigest()!=config['prepared_file_sha256']:
        raise ValueError('Prepared data changed')
    p=_load_prepared(folder);target=make_target(p,config['backend'])
    # Compilation/cache load and disposable warmup precede resumed timing.
    state=load_state(folder/'run-0'/'initial.npz',provenance=config['provenance'])
    before=time.perf_counter();target(state.halves[0]);_sync(p.y.device)
    (folder/f'resume-warmup-{time.time_ns()}.json').write_text(json.dumps({'compile_cache_and_disposable_warmup_seconds':time.perf_counter()-before})+'\n')
    return _continue(folder,p,config,target,config['warmup']+retained_sweeps,started,wall_seconds)


def _verify_config(config):
    values={k:v for k,v in config.items() if k!='provenance'}
    if hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()!=config['provenance']:
        raise ValueError('Saved fit configuration changed')


def _runtime(backend):
    names=['numpy','scipy','torch']+(['numba'] if backend=='numba' else [])
    return {**{name:importlib.metadata.version(name) for name in names},'platform':platform.platform(),'machine':platform.machine()}


@contextmanager
def _threads(backend,torch_threads,numba_threads):
    old_torch=torch.get_num_threads();old_numba=None
    try:
        torch.set_num_threads(torch_threads)
        if backend=='numba':
            import numba
            old_numba=numba.get_num_threads();numba.set_num_threads(numba_threads)
        yield
    finally:
        torch.set_num_threads(old_torch)
        if old_numba is not None:numba.set_num_threads(old_numba)


def sample_parameters(prepared,*,torch_threads=1,numba_threads=2,**options):
    """Fit four or more independent ensembles with controlled CPU threading.

    See fit_grid for preparation and _sample_parameters for explicit stored
    settings. output_dir is required. A resumed target is an absolute retained
    sweep count; no warmup is repeated and no boundary draws are duplicated.
    """
    with _threads(options.get('backend','numba'),torch_threads,numba_threads):
        return _sample_parameters(prepared,**options)


def resume_fit(output_dir,*,retained_sweeps,wall_seconds=900):
    config=json.loads((Path(output_dir)/'config.json').read_text());_verify_config(config)
    with _threads(config['backend'],config['threads']['torch'],config['threads']['numba']):
        return _resume_fit(output_dir,retained_sweeps=retained_sweeps,wall_seconds=wall_seconds)


def load_fit(output_dir):
    """Read historical output without promising continuation after code changes."""
    folder=Path(output_dir).resolve();config=json.loads((folder/'config.json').read_text());_verify_config(config)
    if hashlib.sha256((folder/'prepared.npz').read_bytes()).hexdigest()!=config['prepared_file_sha256']:
        raise ValueError('Prepared data changed')
    return FitResult(folder,_load_prepared(folder),config)


def fit_grid(data,regressors=None,*,uncertainty=None,model='vanilla_ar1',dtype='float64',device='cpu',
             priors=None,sampling='monthly',**options):
    """Labeled independent tensor fit. Parameters first; reconstruct after diagnostics.

    Existing dlmmc.fit remains the compatible CmdStan entry point. This API uses
    explicit affine runs/walkers and a bounded checkpoint directory.
    """
    p=prepare_data(data,regressors,uncertainty=uncertainty,model=model,dtype=dtype,device=device,priors=priors,sampling=sampling)
    return sample_parameters(p,**options)
