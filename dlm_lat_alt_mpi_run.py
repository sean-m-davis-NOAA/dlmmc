# Import stuff
try:
    import pystan
    HAVE_PYSTAN = True
except Exception:
    HAVE_PYSTAN = False
import numpy as np
import sys
import scipy.interpolate as interpolate
import netCDF4
import tqdm
from mpi4py import MPI
import pickle
from utils.utils import *

# Results directory and run name
results_dir = 'results'
results_filename = 'BASIC_V1_2017'

# How many MCMC chains/samples to do
n_chains = 1
warmup = 1
iterations = 3
n_samples = n_chains*(iterations-warmup)

# MPI comm, number of processes and rank info
comm = MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()

# Import the DLM model
# The pickled file may be a PyStan-compiled model (legacy) or a marker dict produced by
# compile_stan_models.py indicating a stan file to compile with cmdstanpy at runtime.
model_obj = pickle.load(open('models/dlm_vanilla_ar1.pkl', 'rb'))
if isinstance(model_obj, dict) and model_obj.get('backend') == 'cmdstanpy':
    model_kalman_ar1 = model_obj
else:
    model_kalman_ar1 = model_obj

# Import the data

# Import data from a netCDF
data = netCDF4.Dataset('data/BASIC_V1_2017_lotus_seascyc_gcsw2017_fac2.nc')

# Extract time, pressure and latitude variables from the netCDF
T = data['time'][:]
P = data['pressure'][:]
L = data['latitude'][:]

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

# Make the netcdf file for saving results
if myrank == 0:
    create_results_netcdf(results_dir, results_filename, L, P, T, n_samples, nregs)

# MPI gridding: meshgrid for pressures and latitudes
LP = np.meshgrid(np.arange(0, len(P)), np.arange(0, len(L)))

# MPI set-up
total_length = len(L)*len(P)
indicies = np.where((np.arange(total_length)%nprocs)==myrank)[0]

# Set up the progress bar
pbar = tqdm.tqdm(total = len(indicies), desc = "press/lats")

# Loop over latitudes and pressures
for ind in indicies:
    
    # Latitude and pressure for this iteration
    latitude = LP[1].flatten()[int(ind)]
    pressure = LP[0].flatten()[int(ind)]

    # Set the data and initialization of parameters that are fed into the DLM

    # Pick out a pressure and latitude panel: this is the "data" time-series from the netCDF
    d = data['o3'][:, pressure, latitude]

    # Extract the error-bars on the time-series from the netCDF
    s = data['o3_sigma'][:, pressure, latitude]
    
    # Prepare missing (NaN) values
    d, s = prepare_missing_data(d, s)
    
    # Check if the data are there
    if np.mean(d) != 0:

        # Regressors: stack of all the regressors together in a 2d array
        regressors = np.column_stack([enso, solar, qbo30, qbo50, saod[latitude]])

        # Data: this is a dictionary of all of the data/inputs that the DLM model needs (descriptions below)
        input_data = {
                        'time_series':d, # float[N] data vector
                        'stddev':s, # float[N] std-dev error bars
                        'N':N, # (int) number of time-steps in the time-series
                        'nreg':nregs, # (int) number of regressors
                        'regressors':regressors, # float[N, nreg] the regressors
                        'sampling':sampling_rate("monthly"), # sampling rate of the data, must be "daily", "monthly" or "annual"
                        'S':10., # prior variance on the regression coefficients (priors are zero mean Gaussian with variance S)
                        'sigma_trend_prior':1e-4, # std-dev of the half-Gaussian prior on sigma_trend that controls how wiggly the trend can be
                        'sigma_seas_prior':0.01, # std-dev of the half-Gaussian prior on sigma_seas that controls how dynamic the seaonal cycle can be
                        'sigma_AR_prior':0.5 # std-dev of the half_Gaussian prior on the AR1 process std-dev
                    }

        # Initialization: Initial guess values for the hyper-parameters
        initial_state = {
                         'sigma_trend':0.0001,
                         'sigma_seas':0.001,
                         'sigma_AR':0.01,
                         'rhoAR1':0.1,
                        }

        # Run the model (supports both PyStan and cmdstanpy marker produced by compile_stan_models.py)
        with suppress_stdout_stderr():
            if isinstance(model_kalman_ar1, dict) and model_kalman_ar1.get('backend')=='cmdstanpy':
                try:
                    from cmdstanpy import CmdStanModel
                except Exception:
                    raise ImportError("cmdstanpy is required for cmdstanpy backend. Install via pip: pip install cmdstanpy")

                # Compile/run with cmdstanpy
                model = CmdStanModel(stan_file=model_kalman_ar1['stan_file'])
                iter_sampling = max(1, iterations - warmup)
                mcmc = model.sample(data=input_data, chains=n_chains, iter_sampling=iter_sampling, iter_warmup=warmup)

                # Convert CmdStanMCMC to a pyStan-like fit with extract()
                try:
                    # Preferred: use draws_pd() to build arrays for each variable
                    df = mcmc.draws_pd()
                    import re, numpy as _np
                    var_dict = {}
                    for col in df.columns:
                        m = re.match(r'(?P<name>[^\[]+)(\[(?P<index>.+)\])?', col)
                        name = m.group('name')
                        if name not in var_dict:
                            var_dict[name] = []
                        var_dict[name].append(df[col].values)
                    extract_dict = {}
                    for name, arrs in var_dict.items():
                        arr = _np.vstack(arrs).T
                        if arr.shape[1] == 1:
                            extract_dict[name] = arr[:,0]
                        else:
                            extract_dict[name] = arr
                except Exception:
                    # Fallback: try stan_variable for a list of expected names
                    extract_dict = {}
                    for name in ('sigma_trend','sigma_seas','sigma_AR','rhoAR1','trend','slope','beta','seasonal'):
                        try:
                            extract_dict[name] = mcmc.stan_variable(name)
                        except Exception:
                            pass

                class FitShim:
                    def __init__(self, extract_dict):
                        self._extract = extract_dict
                    def extract(self):
                        return self._extract

                fit = FitShim(extract_dict)
            else:
                fit = model_kalman_ar1.sampling(data=input_data, iter=iterations, warmup=warmup, chains=n_chains, init = [initial_state for i in range(n_chains)], verbose=False, pars=('sigma_trend', 'sigma_seas', 'sigma_AR', 'rhoAR1', 'trend', 'slope', 'beta', 'seasonal'))

        # Put the relevant bits into the netCDF...
        save_results(results_dir, results_filename, fit, pressure, latitude)

    # Update the progress bar
    pbar.update(1)

# Send signals once through the loop
if comm.rank > 0:
    comm.send(['done.'], 0, tag=myrank)

# Wait until all signals received by rank == 0 process
if myrank == 0:
    for i in range(1, nprocs):
        signals = comm.recv(source=i)

    # Now convert to netCDF
    convert_to_netcdf(results_dir, results_filename, P, L)
