import os
import numpy as np
import scipy.interpolate as interpolate
import xarray as xr
import netCDF4
import tqdm
from mpi4py import MPI
from cmdstanpy import CmdStanModel
from stan_backend import load_model_object
from utils.utils import prepare_missing_data, sampling_rate


def create_results_file(path, time, pressure, latitude, n_samples, nregs):
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as output:
        output.createDimension('time', len(time))
        output.createDimension('pressure', len(pressure))
        output.createDimension('latitude', len(latitude))
        output.createDimension('nsamples', n_samples)
        output.createDimension('nregs', nregs)

        for name, values in (
            ('time', time),
            ('pressure', pressure),
            ('latitude', latitude),
            ('nsamples', np.arange(n_samples)),
            ('nregs', np.arange(nregs)),
        ):
            variable = output.createVariable(name, np.asarray(values).dtype, (name,))
            variable[:] = values

        cell_dims = ('pressure', 'latitude')
        sample_cell_dims = ('nsamples', 'pressure', 'latitude')
        time_cell_dims = ('pressure', 'latitude', 'time')
        sample_time_cell_dims = ('nsamples', 'pressure', 'latitude', 'time')
        sample_reg_dims = ('nsamples', 'pressure', 'latitude', 'nregs')
        variables = {
            'trend': sample_time_cell_dims,
            'trend_mean': time_cell_dims,
            'trend_std': time_cell_dims,
            'slope_mean': time_cell_dims,
            'slope_std': time_cell_dims,
            'seasonal_mean': time_cell_dims,
            'seasonal_std': time_cell_dims,
            'regressor_coefficients': sample_reg_dims,
            'sigma_trend': sample_cell_dims,
            'sigma_seas': sample_cell_dims,
            'sigma_AR': sample_cell_dims,
            'rhoAR1': sample_cell_dims,
        }
        for name, dims in variables.items():
            output.createVariable(name, 'f8', dims, zlib=True, fill_value=np.nan)

        status = output.createVariable('fit_status', 'i1', cell_dims)
        status[:] = 0
        status.flag_values = np.array([-1, 0, 1], dtype=np.int8)
        status.flag_meanings = 'failed skipped complete'
        output.backend = 'cmdstanpy'
        output.writer = 'MPI rank 0; worker results streamed as fits complete'


def write_fit_result(output, pressure, latitude, draws, error=None):
    if error is not None:
        output['fit_status'][pressure, latitude] = -1
        print(f'Fit failed at pressure index {pressure}, latitude index {latitude}: {error}')
        return
    if draws is None:
        output['fit_status'][pressure, latitude] = 0
        return

    trend = draws['trend']
    output['trend'][:, pressure, latitude, :] = trend
    output['trend_mean'][pressure, latitude, :] = np.mean(trend, axis=0)
    output['trend_std'][pressure, latitude, :] = np.std(trend, axis=0)
    for name in ('slope', 'seasonal'):
        output[f'{name}_mean'][pressure, latitude, :] = np.mean(draws[name], axis=0)
        output[f'{name}_std'][pressure, latitude, :] = np.std(draws[name], axis=0)
    output['regressor_coefficients'][:, pressure, latitude, :] = draws['beta']
    for name in ('sigma_trend', 'sigma_seas', 'sigma_AR', 'rhoAR1'):
        output[name][:, pressure, latitude] = draws[name]
    output['fit_status'][pressure, latitude] = 1

# Results directory and run name
results_dir = 'results'
results_filename = 'BASIC_V1_2017'

# How many MCMC chains/samples to do
n_chains = 1
warmup = 1
iterations = 3
n_samples = n_chains * (iterations - warmup)

# MPI comm, number of processes and rank info
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

# Load the CmdStan model marker created by compile_stan_models.py. Rank 0 compiles
# once, then the other ranks open the resulting executable.
model_marker = load_model_object('models/dlm_vanilla_ar1.pkl')
if not isinstance(model_marker, dict) or model_marker.get('backend') != 'cmdstanpy':
    raise RuntimeError(
        'Expected a CmdStanPy model marker. Run compile_stan_models.py with cmdstanpy installed.'
    )
stan_file = model_marker['stan_file']
compile_error = None
model = None
if myrank == 0:
    try:
        model = CmdStanModel(stan_file=stan_file)
    except Exception as exc:
        compile_error = str(exc)
compile_error = comm.bcast(compile_error, root=0)
if compile_error:
    raise RuntimeError(f'Could not compile CmdStan model: {compile_error}')
if myrank != 0:
    model = CmdStanModel(stan_file=stan_file)
comm.Barrier()

# Open the input dataset with named xarray dimensions and coordinates.
data = xr.open_dataset('data/BASIC_V1_2017_lotus_seascyc_gcsw2017_fac2.nc')
T = data['time'].to_numpy()
P = data['pressure'].to_numpy()
L = data['latitude'].to_numpy()

# How many time steps are there? (ie how long is the time-series)
N = len(T)

# Import the regressors and project them onto the time grid corresponding to the imported data

# ENSO
regressor_data = np.loadtxt('regressors/ENSO_MEI_1950_201802.txt')
Y = interpolate.InterpolatedUnivariateSpline(regressor_data[:,0], regressor_data[:,1])
enso = Y(T)

# SOLAR
regressor_data = np.loadtxt('regressors/Flux_F30_monthly_195111_201803_absolute.txt')
Y = interpolate.InterpolatedUnivariateSpline(regressor_data[:,0], regressor_data[:,1])
solar = Y(T)

# QBO30
regressor_data = np.loadtxt('regressors/multi_qbo30_1953_2018.txt')
Y = interpolate.InterpolatedUnivariateSpline(regressor_data[:,0], regressor_data[:,1])
qbo30 = Y(T)

# QBO50
regressor_data = np.loadtxt('regressors/multi_qbo50_1953_2018.txt')
Y = interpolate.InterpolatedUnivariateSpline(regressor_data[:,0], regressor_data[:,1])
qbo50 = Y(T)

# SAOD
regressor_data = np.loadtxt('regressors/sad_1979_2017_10deg_60S_60N.txt')
saod = [0]*12
for latitude in range(12):
    Y = interpolate.InterpolatedUnivariateSpline(regressor_data[:,0], regressor_data[:,latitude+1])
    saod[latitude] = Y(T)

# How many regressors?
nregs = 5

def fit_cell(pressure, latitude):
    d = data['o3'].isel(pressure=pressure, latitude=latitude).to_numpy()
    s = data['o3_sigma'].isel(pressure=pressure, latitude=latitude).to_numpy()
    d, s = prepare_missing_data(d, s)
    if np.mean(d) == 0:
        return None

    regressors = np.column_stack([enso, solar, qbo30, qbo50, saod[latitude]])
    input_data = {
        'time_series': d,
        'stddev': s,
        'N': N,
        'nreg': nregs,
        'regressors': regressors,
        'sampling': sampling_rate('monthly'),
        'S': 10.0,
        'sigma_trend_prior': 1e-4,
        'sigma_seas_prior': 0.01,
        'sigma_AR_prior': 0.5,
    }
    fit = model.sample(
        data=input_data,
        chains=n_chains,
        iter_sampling=iterations - warmup,
        iter_warmup=warmup,
        show_progress=False,
    )
    return {
        name: fit.stan_variable(name)
        for name in (
            'trend', 'slope', 'seasonal', 'beta', 'sigma_trend',
            'sigma_seas', 'sigma_AR', 'rhoAR1'
        )
    }


total_length = len(L) * len(P)
indices = np.arange(total_length)
if nprocs == 1:
    local_indices = indices
elif myrank == 0:
    local_indices = np.empty(0, dtype=int)
else:
    local_indices = indices[indices % (nprocs - 1) == myrank - 1]

output_path = os.path.join(results_dir, f'{results_filename}.nc')
if myrank == 0:
    os.makedirs(results_dir, exist_ok=True)
    create_results_file(output_path, T, P, L, n_samples, nregs)
comm.Barrier()

if myrank == 0:
    pbar = tqdm.tqdm(total=total_length, desc='fits written')
    with netCDF4.Dataset(output_path, 'r+') as output:
        if nprocs == 1:
            for ind in local_indices:
                latitude = int(ind // len(P))
                pressure = int(ind % len(P))
                try:
                    draws = fit_cell(pressure, latitude)
                    error = None
                except Exception as exc:
                    draws = None
                    error = str(exc)
                write_fit_result(output, pressure, latitude, draws, error)
                pbar.update(1)
        else:
            for _ in range(total_length):
                pressure, latitude, draws, error = comm.recv(
                    source=MPI.ANY_SOURCE,
                    tag=1,
                )
                write_fit_result(output, pressure, latitude, draws, error)
                pbar.update(1)
    pbar.close()
else:
    for ind in local_indices:
        latitude = int(ind // len(P))
        pressure = int(ind % len(P))
        try:
            draws = fit_cell(pressure, latitude)
            error = None
        except Exception as exc:
            draws = None
            error = str(exc)
        comm.send((pressure, latitude, draws, error), dest=0, tag=1)

comm.Barrier()
data.close()
