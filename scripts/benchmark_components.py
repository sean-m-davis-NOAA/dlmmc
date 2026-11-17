"""Bounded before/after API benchmark; independent frozen panels only."""
from pathlib import Path
import argparse,json,time,hashlib,platform
import numpy as np
import torch
from dataclasses import replace
import dlmmc
from dlmmc.examples import load_panels
from dlmmc.sampling import make_target
from dlmmc.structured_mps import compile_kernel
from dlmmc.mps_smoothing import compile_smoother
from dlmmc.production import initial_center
parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--form',choices=['preset','composed'],required=True);parser.add_argument('--kind',choices=['fixed','fit'],required=True);a=parser.parse_args()
start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
y,s,x,_=load_panels(a.repo/'examples/modernization');assert y.sizes['panel']==5
record={'form':a.form,'kind':a.kind,'source':dlmmc.__file__,'platform':platform.platform(),'torch':torch.__version__,'device':'native mps','dtype':'float32','panel_ids':list(map(str,y.panel.values)),'seed':2026111701,'models':[]}
model=lambda name:dlmmc.Model.from_preset(name) if a.form=='composed' else name
sync=torch.mps.synchronize
arrays={}
if a.kind=='fixed':
    for name in ['vanilla_ar1','vanilla_ar2','noregs_ar1','dynregs_ar1','vanilla_ar1_noseasonal','vanilla_ar2_noseasonal']:
        yy,xx=y,x
        if 'noseasonal' in name:
            yy=dlmmc.deseasonalize_harmonics(y).deseasonalized;xx=dlmmc.deseasonalize_harmonics(x).deseasonalized
        if name=='noregs_ar1':xx=None
        p=dlmmc.prepare_data(yy,xx,uncertainty=s,model=model(name),dtype='float32',device='mps')
        rng=np.random.default_rng(2026111701)
        u=torch.tensor(initial_center(p)+rng.normal(0,.06,(64,5,len(p.parameter_names))),dtype=torch.float32,device='mps')
        target=make_target(p,'mps');sync();begin=time.perf_counter();compile_kernel(p.model,p.regressors.shape[-1]);compile_smoother(p.model,p.regressors.shape[-1]);sync();compilation=time.perf_counter()-begin
        begin=time.perf_counter();target(u);mom=dlmmc.conditional_trend_moments(p,u[:8],156,backend='mps');sync();warmup=time.perf_counter()-begin
        row={'name':name,'parameters':len(p.parameter_names),'compilation_or_cache_load_seconds':compilation,'disposable_exact_shape_warmup_seconds':warmup}
        for operation,callback,count in [('likelihood',lambda:target(u),25),('smoother',lambda:dlmmc.conditional_trend_moments(p,u[:8],156,backend='mps'),10)]:
            timings=[]
            for _ in range(count):
                sync();begin=time.perf_counter();value=callback();sync();timings.append(time.perf_counter()-begin)
            row[operation+'_seconds']=timings;arrays[name+'_'+operation]=value.cpu().numpy()
        record['models'].append(row);print(name,'likelihood',np.median(row['likelihood_seconds']),'smoother',np.median(row['smoother_seconds']),flush=True)
        if time.perf_counter()-start>300:raise TimeoutError('Fixed benchmark actual cap')
    np.savez(a.output/'outputs.npz',**arrays)
else:
    before=time.perf_counter()
    fit=dlmmc.fit_production(y,x,uncertainty=s,model=model('vanilla_ar1'),backend='mps',output_dir=a.output/'fit',seed=2026111701,wall_seconds=600)
    sync();fitwall=time.perf_counter()-before
    diag=dlmmc.diagnose_production(fit)
    before=time.perf_counter();summary=dlmmc.summarize_production_trends(fit,output_dir=a.output/'trends',anchor_month='1998-01',wall_seconds=300);sync();outputwall=time.perf_counter()-before
    comp=json.loads((a.output/'fit/compilation.json').read_text());init=json.loads((a.output/'fit/initialization.json').read_text())
    sampling=0
    for path in (a.output/'fit/run-0').glob('block-*.npz'):
        with np.load(path) as z:sampling+=float(z['sampling_seconds'])
    record.update(profile=dlmmc.PRODUCTION_PROFILE.to_dict(),fit_wall_seconds=fitwall,output_wall_seconds=outputwall,compilation=comp,initialization=init,sampling_seconds=sampling,output=summary,monitoring=diag)
    record['fit_excluding_compilation_warmup_seconds']=fitwall-comp['compilation_or_cache_load_seconds']-comp['disposable_exact_shape_warmup_seconds']
    record['end_to_end_excluding_compilation_warmup_seconds']=record['fit_excluding_compilation_warmup_seconds']+outputwall-summary['compilation_or_cache_load_seconds']-summary['disposable_exact_shape_warmup_seconds']
    if time.perf_counter()-start>900:raise TimeoutError('Fit+output benchmark actual cap')
record['actual_process_seconds']=time.perf_counter()-start
record['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(dlmmc.__file__).parent.glob('*.py')}
(a.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(a.output,'completed',record['actual_process_seconds'],flush=True)
