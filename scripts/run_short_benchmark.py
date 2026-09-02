#!/usr/bin/env python3
import time, json, os,traceback, sys, pathlib
from time import perf_counter
import pickle
import numpy as np
import netCDF4
# make repo root importable
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from utils.utils import prepare_missing_data, sampling_rate
from datetime import datetime

def main():
    out = {}
    out['timestamp'] = datetime.utcnow().isoformat()+"Z"
    try:
        model_pkl = 'models/dlm_vanilla_ar1.pkl'
        model_obj = None
        if os.path.exists(model_pkl):
            try:
                model_obj = pickle.load(open(model_pkl,'rb'))
                out['model_pickle_type'] = type(model_obj).__name__
            except Exception as e:
                out['model_pickle_load_error'] = str(e)
        ds_path = 'data/BASIC_V1_2017_lotus_seascyc_gcsw2017_fac2.nc'
        if not os.path.exists(ds_path):
            raise RuntimeError('data file not found: '+ds_path)
        ds = netCDF4.Dataset(ds_path)
        data_var = None
        data_name = None
        for name, var in ds.variables.items():
            if name.lower() in ('time','latitude','longitude','lat','lon','pressure','plev'):
                continue
            if 'time' in var.dimensions:
                data_var = var
                data_name = name
                break
        if data_var is None:
            raise RuntimeError('no suitable data variable found')
        idx = []
        for dim in data_var.dimensions:
            if dim == 'time': idx.append(slice(None))
            else: idx.append(0)
        arr = data_var[tuple(idx)]
        d = np.array(arr, dtype=float).squeeze()
        s = None
        for cand in ('stddev','std','error','uncertainty','sigma'):
            if cand in ds.variables:
                v = ds.variables[cand]
                if 'time' in v.dimensions:
                    s = np.array(v[tuple(idx)], dtype=float).squeeze(); break
        if s is None:
            s = np.ones_like(d) * 1e-3
        d, s = prepare_missing_data(d, s)
        input_data = {
            'time_series': d.tolist(),
            'stddev': s.tolist(),
            'N': int(len(d)),
            'nreg': 1,
            'regressors': [[0.0] for _ in range(len(d))],
            'sampling': float(sampling_rate('monthly')),
            'S': 10.0,
            'sigma_trend_prior':1e-4,
            'sigma_seas_prior':0.01,
            'sigma_AR_prior':0.5
        }
        out['settings'] = dict(warmup=50, sampling=200, chains=4)
        from cmdstanpy import CmdStanModel
        stan_file = 'models/dlm_vanilla_ar1.stan'
        print('Compiling model from', stan_file)
        t0 = perf_counter()
        model = CmdStanModel(stan_file=stan_file)
        out['compile_time'] = perf_counter() - t0
        print('Compile time:', out['compile_time'])
        print('Starting sampling')
        t0 = perf_counter()
        fit = model.sample(data=input_data, chains=4, iter_warmup=50, iter_sampling=200)
        out['sample_time'] = perf_counter() - t0
        print('Sample time:', out['sample_time'])
        try:
            df = fit.draws_pd()
            if 'divergent__' in df.columns:
                out['n_divergent'] = int(df['divergent__'].sum())
            else:
                out['n_divergent'] = None
        except Exception as e:
            out['draws_error'] = str(e)
        try:
            out['summary'] = fit.summary().to_dict()
        except Exception as e:
            out['summary_error'] = str(e)
    except Exception as e:
        out['error'] = traceback.format_exc()
    os.makedirs('results', exist_ok=True)
    outfile = f'results/benchmark_short_{int(time.time())}.json'
    with open(outfile,'w') as f:
        json.dump(out, f, indent=2)
    print('WROTE', outfile)

if __name__=='__main__':
    main()
