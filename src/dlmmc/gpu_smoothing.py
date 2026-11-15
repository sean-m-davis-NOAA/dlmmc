"""GPU32 trend moments, bounded histories and device-side mixture reduction."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from .production import production_parameter_cache
from .sampling import _sync
from .diagnostics import autocorrelation_time
from .compute import guarded_gpu_arithmetic


@torch.no_grad()
def fixed_cuda_trend_moments(p,u,anchor,*,regressors=None):
    """Portable Torch CUDA implementation; native CUDA qualification pending."""
    from .tensor_target import model_components
    from .precision_prototype import filter_tensor
    from .tensor_smoothing import conditional_moments
    F,G,Q,C,_,valid=model_components(u,p.y,p.variance,p.observed,p.regressors if regressors is None else regressors,p.model,S=p.priors['S'],sampling=p.sampling)
    if not bool(valid.all()):raise ValueError('Invalid selected parameter; no removal')
    filtered=filter_tensor(*p.arguments[:3],F,G,Q,C)
    m,c,j=conditional_moments(filtered,G,Q);n=m.shape[-2]
    variance=c[...,0,0];change=m[...,0]-m[...,anchor,0,None]
    cv=torch.empty_like(change);cv[...,anchor]=0
    cross=c[...,anchor,:,0]
    for t in range(anchor-1,-1,-1):
        cross=(j[...,t,:,:]@cross.unsqueeze(-1)).squeeze(-1)
        cv[...,t]=variance[...,t]+variance[...,anchor]-2*cross[...,0]
    weight=torch.zeros_like(m[...,0,:]);weight[...,0]=1
    for t in range(anchor+1,n):
        weight=(j[...,t-1,:,:].transpose(-1,-2)@weight.unsqueeze(-1)).squeeze(-1)
        cv[...,t]=variance[...,t]+variance[...,anchor]-2*(weight*c[...,t,:,0]).sum(-1)
    result=torch.stack((m[...,0],variance,change,cv),-1)
    if not bool(torch.isfinite(result).all()) or bool((result[...,1::2]<0).any()):raise FloatingPointError('Invalid GPU covariance; no floor')
    scale=u.new_tensor(p.scale)[None,:,None];offset=u.new_tensor(p.offset)[None,:,None]
    result[...,0]=result[...,0]*scale+offset;result[...,1]*=scale*scale;result[...,2]*=scale;result[...,3]*=scale*scale
    return result


def select_draws(fit,per_sweep,seed,retained_sweeps=None):
    u=production_parameter_cache(fit)
    if retained_sweeps is not None:
        if retained_sweeps<8 or retained_sweeps>u.shape[1] or retained_sweeps%2:raise ValueError('Valid even retained prefix of at least eight steps required')
        u=u[:,:retained_sweeps]
    runs,steps,walkers,cells,_=u.shape
    if runs!=1 or steps<8 or steps%2 or not 1<=per_sweep<=walkers:raise ValueError('One ensemble, at least eight even retained steps and valid draw selection required')
    selected=np.empty((steps*per_sweep,cells,u.shape[-1]),np.float32);selections=[]
    for g in range(cells):
        rng=np.random.default_rng(np.random.SeedSequence([seed,g,0]));count=min(walkers,max(4,per_sweep))
        ww=np.stack([rng.choice(walkers,count,replace=False) for _ in range(steps)])[:,:per_sweep]
        selected[:,g]=u[0,np.repeat(np.arange(steps),per_sweep),ww.ravel(),g];selections.append(ww)
    return selected,np.stack(selections),steps,walkers


@torch.no_grad()
@guarded_gpu_arithmetic
def summarize_gpu_trends(fit,*,output_dir,anchor_month='1998-01',selected_walkers_per_sweep=None,
        seed=20261116,wall_seconds=5400,workspace_bytes=1024**3,retained_sweeps=None):
    started=time.monotonic();deadline=started+wall_seconds;p=fit.prepared;folder=Path(output_dir)
    if p.y.device.type not in ('mps','cuda') or p.y.dtype!=torch.float32:raise ValueError('Explicit GPU32 fit required')
    if folder.exists():raise ValueError('Preserve previous output')
    folder.mkdir(parents=True)
    matches=np.flatnonzero(p.time.astype('datetime64[M]')==np.datetime64(anchor_month,'M'))
    if len(matches)!=1:raise ValueError('Unique anchor month required')
    anchor=int(matches[0]);per=selected_walkers_per_sweep if selected_walkers_per_sweep is not None else fit.config["execution_profile"].get("output_draws_per_sweep",4)
    selected,selections,steps,walkers=select_draws(fit,per,seed,retained_sweeps);draws,cells,_=selected.shape;n=len(p.time)
    selection_seconds=time.monotonic()-started
    from .structured_mps import state_layout
    d,_=state_layout(p.model,p.regressors.shape[-1]);history=2*d*d+d
    if p.y.device.type=='mps':
        from .mps_smoothing import fixed_mps_trend_moments,compile_smoother
        scalar_workspace=history+4
        operation=lambda u:fixed_mps_trend_moments(p,u,anchor,workspace_bytes=workspace_bytes)
    else:
        scalar_workspace=6*d*d+3*d
        operation=lambda u:fixed_cuda_trend_moments(p,u,anchor)
    tile=min(draws,max(0,int(workspace_bytes)//(cells*n*scalar_workspace*4)))
    if tile<1:raise ValueError('Workspace must fit at least one draw for all panels')
    # A selected sweep can span several draw tiles without splitting panels.
    tile=tile//per*per if tile>=per else max(d for d in range(1,tile+1) if per%d==0)
    before=time.perf_counter();z=torch.tensor(selected,device=p.y.device,dtype=torch.float32);_sync(p.y.device)
    preparation=time.perf_counter()-before
    before=time.perf_counter()
    if p.y.device.type=='mps':_,kernel_seconds=compile_smoother(p.model,p.regressors.shape[-1])
    else:kernel_seconds=0.
    compilation=time.perf_counter()-before
    def evaluate(values,start,label):
        try:return operation(values)
        except BaseException:
            np.savez(folder/f'failed-{label}-{start}-{time.time_ns()}.npz',u=values.cpu().numpy(),anchor=anchor,
                prepared_checksum=p.checksum,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            raise
    before=time.perf_counter()
    # Each exact active tile shape is warmed once. No draw-dependent compilation.
    shapes=sorted({tile,draws%tile}-{0})
    for size in shapes:evaluate(z[:size],0,'warmup');_sync(p.y.device)
    warm=time.perf_counter()-before
    if time.monotonic()>=deadline:raise TimeoutError('GPU output warmup cap')
    print(f'GPU smoother: {draws} draws x {cells} panels; tile={tile} draws, all panels; warmup {warm:.3f}s',flush=True)
    before=time.perf_counter();mean=z.new_zeros((cells,n,2));m2=torch.zeros_like(mean);conditional=torch.zeros_like(mean)
    halves=z.new_zeros((2,cells,n));series=z.new_zeros((steps,cells,3));indices=[0,n//2,n-1];count=0
    for start in range(0,draws,tile):
        if time.monotonic()>=deadline:raise TimeoutError('GPU output cap; retained sampler checkpoints unchanged')
        end=min(draws,start+tile);moments=evaluate(z[start:end],start,'retained');amount=end-start
        values=moments[...,::2];local=values.mean(0);delta=local-mean
        m2+=((values-local)**2).sum(0)+delta**2*(count*amount/(count+amount));mean+=delta*(amount/(count+amount))
        conditional+=moments[...,1::2].sum(0)
        count+=amount
        mid=draws//2
        if start<mid:halves[0]+=moments[:min(amount,mid-start),...,0].sum(0)/(draws//2)
        if end>mid:halves[1]+=moments[max(0,mid-start):,...,0].sum(0)/(draws//2)
        if amount>=per:
            series[start//per:end//per]=moments[:,:,:,0].reshape(amount//per,per,cells,n).mean(1)[:,:,indices]
        else:
            series[start//per]+=moments[:,:,:,0].sum(0)[:,indices]/per
        if start==0 or end==draws or (start//tile+1)%64==0:
            _sync(p.y.device);print(f'GPU smooth {end}/{draws} draws; {time.perf_counter()-before:.2f}s',flush=True)
    variance=(m2+conditional)/draws
    if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all() and (variance>=0).all()):raise FloatingPointError('Invalid mixture output; no floor')
    result=torch.stack((mean[...,0],torch.sqrt(variance[...,0]),mean[...,1],torch.sqrt(variance[...,1])),-1)
    _sync(p.y.device);compute=time.perf_counter()-before
    before=time.perf_counter();result=result.cpu().numpy();halves=halves.cpu().numpy();series=series.cpu().numpy().transpose(1,0,2);rows=[]
    for g in range(cells):
        mu,sd,change,change_sd=result[g].T;tau=np.array([autocorrelation_time(series[g,:,k]) for k in range(3)])
        mcse=np.sqrt(series[g].astype(float).var(0,ddof=1)*tau/steps);flags=['SINGLE_ENSEMBLE_OUTPUT_MCSE_PROVISIONAL']
        if np.any(steps<50*tau):flags.append('SHORT_OUTPUT_CORRELATION_ESTIMATE')
        if np.any(mcse/sd[indices]>.05):flags.append('OUTPUT_MEAN_MCSE_OVER_0.05_POSTERIOR_SD')
        path=folder/f'cell-{g:04d}.npz'
        np.savez(path,mean=mu,sd=sd,change_mean=change,change_sd=change_sd,time=p.time,
            selected_walker=selections[g][None],mcse_time_indices=np.array(indices),selected_time_ensemble_mean=series[g][None],
            first_half_mean=halves[0,g],second_half_mean=halves[1,g])
        row=dict(cell=g,panel=str(p.cell_coordinates['panel'][g]),flags=flags,mcse_indices=indices,mean_mcse=mcse.tolist(),
            mcse_over_posterior_sd=(mcse/sd[indices]).tolist(),maximum_iat=float(tau.max()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (folder/f'cell-{g:04d}.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    io=time.perf_counter()-before
    record=dict(status='PRODUCTION_TREND_ESTIMATES_WITH_FLAGS',qualification_eligible=False,cells=rows,panels=cells,
        device=str(p.y.device),dtype='float32',anchor_month=anchor_month,anchor_index=anchor,fit_provenance=fit.config['provenance'],
        seed=seed,retained_prefix=retained_sweeps,selected_walkers_per_sweep=per,parameter_iterations=[1,steps,walkers,cells],full_batch_shape=[draws,cells],
        panel_chunking=False,draw_chunking=True,draw_tile=tile,workspace_limit_bytes=workspace_bytes,
        history_workspace_bytes=tile*cells*n*(history if p.y.device.type=='mps' else scalar_workspace-4)*4,
        selection_seconds=selection_seconds,preparation_seconds=preparation,compilation_or_cache_load_seconds=compilation,
        kernel_compile_seconds=kernel_seconds,disposable_exact_shape_warmup_seconds=warm,warmed_tile_shapes=shapes,
        smoothing_and_reduction_seconds=compute,diagnostics_and_output_io_seconds=io,
        post_compile_output_seconds=time.monotonic()-started-compilation-warm,actual_wall_seconds=time.monotonic()-started,
        cap_seconds=wall_seconds,cap_exceeded=time.monotonic()>deadline,
        source_identity={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('gpu_smoothing.py','mps_smoothing.py','tensor_smoothing.py')})
    (folder/'execution.json').write_text(json.dumps(record,indent=2)+'\n');return record
