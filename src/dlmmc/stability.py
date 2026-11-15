"""User-approved MC-aware stability; strict diagnostic flags stay separate."""
import numpy as np
from scipy.stats import chi2


def _batch_variance(x, block):
    r,t,w,p=x.shape
    average=x.reshape(r,t//block,block,w,p).mean((2,3))
    v=average.var(1,ddof=1)/(t//block)/r**2
    total=v.sum(0)
    with np.errstate(divide='ignore',invalid='ignore'):
        df=total**2/np.sum(v*v/(t//block-1),axis=0)
    ci=np.stack((df*total/chi2.ppf(.975,df),df*total/chi2.ppf(.025,df)),-1)
    return total,ci


def mc_stability(x, function_reports, *, seed=2026111203, replicates=1999):
    """Whole-ensemble blocks; no discarded suffix or independent walkers.

    Checks each independent run's half/full mean and SD change with paired
    resampling. Returns conservative precision from autocorrelation and two
    batch sizes. Point flags and interval eligibility are separate fields.
    """
    x=np.asarray(x,dtype=float);r,t,w,p=x.shape
    tau=max(np.max(d['iat_by_run']) for d in function_reports.values())
    if not np.isfinite(tau):return dict(eligible=False,status='NONFINITE_AUTOCORRELATION')
    required=max(64,int(np.ceil(5*tau)))
    block=next((b for b in range(required,t//6+1) if t%b==0 and (t//b)%2==0 and b%2==0),None)
    if block is None:return dict(eligible=False,status='INSUFFICIENT_COMPLETE_BLOCKS',required_block_steps=required)
    center=x.mean((0,1,2));functions={'mean':x,'squared_deviation':(x-center)**2}
    for q in (.05,.5,.95):functions[f'cdf_{q}']=(x<=np.quantile(x,q,axis=(0,1,2))).astype(float)
    precision={}
    for name,values in functions.items():
        a,ca=_batch_variance(values,block);b,cb=_batch_variance(values,block//2)
        auto=np.asarray(function_reports[name]['mean_mcse'])**2
        conservative=np.maximum.reduce([a,b,auto])
        with np.errstate(divide='ignore',invalid='ignore'):
            ess=np.minimum(r*t*w,values.var((0,1,2),ddof=1)/conservative)
        overlap=(ca[:,0]<=cb[:,1])&(cb[:,0]<=ca[:,1])
        eligible=bool(np.all(overlap)&np.all(np.isfinite(ess)&(ess>=2000)))
        precision[name]=dict(eligible=eligible,ess=ess.tolist(),mcse=np.sqrt(conservative).tolist(),
            block_variance_ci=ca.tolist(),half_block_variance_ci=cb.tolist(),variance_intervals_overlap=overlap.tolist())
    # Center before summing squared values for stable width estimates. Each
    # half is a stratum; the original pairing of first-half/full is retained.
    nblocks=t//block;half=nblocks//2;size=block*w;n=t*w
    blocks=(x-center).reshape(r,nblocks,size,p)
    sums=blocks.sum(2);squares=(blocks**2).sum(2)
    rng=np.random.default_rng(seed);out=[]
    for run in range(r):
        counts1=rng.multinomial(half,np.full(half,1/half),size=replicates)
        counts2=rng.multinomial(half,np.full(half,1/half),size=replicates)
        sum1=counts1@sums[run,:half];sum2=counts2@sums[run,half:]
        sq1=counts1@squares[run,:half];sq2=counts2@squares[run,half:]
        m1=sum1/(n/2);mf=(sum1+sum2)/n
        v1=(sq1-(n/2)*m1*m1)/(n/2-1)
        vf=(sq1+sq2-n*mf*mf)/(n-1)
        if np.any(v1<=0)|np.any(vf<=0):return dict(eligible=False,status='DEGENERATE_STABILITY_BOOTSTRAP',block_steps=block)
        first=x[run,:t//2].reshape(-1,p);full=x[run].reshape(-1,p)
        scale=full.std(0,ddof=1)
        point=np.stack(((full.mean(0)-first.mean(0))/scale,full.std(0,ddof=1)/first.std(0,ddof=1)-1))
        boot=np.stack(((mf-m1)/scale,np.sqrt(vf/v1)-1),axis=1)
        error=boot-point;quant=np.quantile(error,[.025,.975],axis=0)
        ci=np.stack((point-quant[1],point-quant[0]),-1)
        compatible=(ci[...,0]<=.05)&(ci[...,1]>=-.05)
        out.append(dict(run=run,metrics=['mean_change_in_full_sd','sd_ratio_minus_one'],
            point=point.tolist(),interval=ci.tolist(),point_outside_band=(abs(point)>.05).tolist(),
            compatible=compatible.tolist(),eligible=bool(compatible.all())))
    eligible=all(d['eligible'] for d in precision.values()) and all(d['eligible'] for d in out)
    return dict(eligible=eligible,status='MC_STABILITY_ELIGIBLE' if eligible else 'INCONCLUSIVE',
                block_steps=block,blocks_per_run=nblocks,seed=seed,replicates=replicates,
                minimum_ess=min(min(d['ess']) for d in precision.values()),precision=precision,stability=out)
