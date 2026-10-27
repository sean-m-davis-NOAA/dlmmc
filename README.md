[![GitHub release](https://img.shields.io/github/v/release/sean-m-davis-NOAA/dlmmc)](https://github.com/sean-m-davis-NOAA/dlmmc/releases/latest)
# DLMMC

### Note: This Python 3 package uses CmdStanPy for Stan model fitting.

**Dynamical Linear Modelling** (DLM) regression code in python for analysis of time-series data. The code is targeted at atmospheric time-series analysis, with a detailed worked example (and data) included for stratospheric ozone, but is a fairly general suite of state space model that can be applied or extended to a wide range of problems.

The core of this package is a suite of DLM models implemented in [stan](https://mc-stan.org), using a combination of HMC sampling and Kalman filtering to infer the DLM model parameters (trend, seasonal cycle, auto-regressive processes etc) given some time-series data. To make the code as accessible as possible, I provide a step-by-step tutorial in python for how to read in your data, run the DLM model(s), and process the outputs to make nice plots. Once you've worked through this tutorial you should have all the tools you need to apply DLM to your own data!

Note: some basic working knowledge of python and jupyter notebooks is required to make the most of this package (but it's not too onerous).

## Installation

Clone the repository and navigate into the directory:

```bash
git clone https://github.com/sean-m-davis-NOAA/dlmmc.git
cd dlmmc
```

### Installation with conda (recommended)

The recommended way to get started is with Conda and the CmdStanPy backend.

CmdStan produces native compiled Stan executables, and CmdStanPy provides the Python interface.

```
# create and activate a modern environment
conda create -n dlmmc-cmdstanpy -c conda-forge python=3.10 cmdstanpy netCDF4 numpy scipy xarray arviz tqdm mpi4py openmpi matplotlib jupyter ipykernel pandas -y
conda activate dlmmc-cmdstanpy
python -m pip install -e .
```

Notes:
- CmdStan requires a C++ toolchain to compile models. On macOS install Xcode command-line tools (`xcode-select --install`) or use a conda-provided compiler on CI.
- The CmdStanPy environment command includes the dependencies for the MPI example. On HPC systems, use the site-provided MPI implementation and ensure `mpi4py` is built against it instead of mixing in a separate Conda MPI implementation.
- To keep a macOS laptop awake while compiling/sampling use `caffeinate -i <command>` or run the job on a remote server or in a tmux session.
  
Finally, if you want to see what a successful installation looks like, see [INSTALL.md](INSTALL.md)

## Getting Started

The recommended path below uses the CmdStanPy environment above. Run commands from the repository root.

1. **Install CmdStan.** A C++ toolchain is required; see the installation notes above.

```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

2. **Optionally precompile the packaged models.** Models compile lazily on the first `fit()` call; this command precompiles all models up front.

```bash
python compile_stan_models.py
```

3. **Run the benchmark.** This is the quickest end-to-end check of data loading, model compilation, and CmdStanPy sampling.

```bash
python dlm_benchmark.py
```

The benchmark writes a timestamped JSON file under `results/`. Confirm it contains `compile_seconds`, `fit_seconds`, and posterior dimensions.

4. **Work through the tutorial.** Open `dlm_tutorial.ipynb` in JupyterLab or VS Code, select the CmdStanPy environment as the kernel, and run the cells in order. The tutorial uses four chains with 1,000 warmup and 2,000 sampling iterations per chain, so it takes longer than the benchmark.

5. **Run the validation suite.** Open `dlm_validation_tests.ipynb` and run the cells in order. It performs synthetic parameter-recovery tests for four model variants using short chains. Review the recovery intervals, R-hat/ESS values, and divergences; the short settings demonstrate the workflow and are not a guarantee of convergence.

6. **Optionally test the gridded MPI workflow.** This fits the example pressure/latitude grid in parallel and streams results to one NetCDF writer. It requires a working MPI implementation and `mpi4py`; on a local Open MPI setup:

```bash
mpirun -np 4 python dlm_lat_alt_mpi_run.py
```

The MPI example defaults to very short chains for a pipeline smoke test, not scientific inference. See [MPI-README.md](MPI-README.md) for output verification and HPC launch details.

### Installation with pip

In an existing Python environment with the required dependencies and CmdStan installed, install the package from the repository root with `python -m pip install .`. For editable development installs, use `python -m pip install -e .`.


**Platforms** 

CmdStan requires a supported C++ toolchain on each platform. On HPC systems, follow the site's compiler and MPI guidance in `INSTALL.md` and `MPI-README.md`.

## Usage

Follow the [Getting Started](#getting-started) sequence for setup and examples. A basic no-regressor fit from xarray data looks like this:

```python
import xarray as xr
import dlmmc

with xr.open_dataset("data/BASIC_V1_2017_lotus_seascyc_gcsw2017_fac2.nc") as ds:
	series = ds["o3"].isel(pressure=22, latitude=8)
	uncertainty = ds["o3_sigma"].isel(pressure=22, latitude=8)
	posterior = dlmmc.fit(
		series,
		uncertainty=uncertainty,
		model="noregs_ar1",
		chains=4,
		warmup=500,
		samples=1000,
	)
```

The returned posterior is an xarray Dataset with `chain`, `draw`, and `time` coordinates. For regressors and plots, see `dlm_tutorial.ipynb`. For grid workloads, see `dlm_lat_alt_mpi_run.py` and [MPI-README.md](MPI-README.md).

**Model descriptions**

Mathematical descriptions of each of the DLM models implemented in this package can be found in the file `models/model_descriptions/model_descriptions.pdf`. This file contains a concise description of the parameters of each model, their physical meanings, and how to refer to them in the code: make sure you have read and understand the model description before running a new model!

## Validation

`dlm_validation_tests.ipynb` provides synthetic parameter-recovery demonstrations for four CmdStanPy model variants. Interpret its recovery intervals and sampler diagnostics as evidence for each run, not as a guarantee that every parameter is well identified under all settings.

## Citing this code

There is a JOSS paper describing dlmmc v1, which can be found [here](http://joss.theoj.org/papers/10.21105/joss.01157). Please cite this paper if you use or refer to this code, as:

_Alsing, (2019). dlmmc: Dynamical linear model regression for atmospheric time-series analysis. Journal of Open Source Software, 4(37), 1157, https://doi.org/10.21105/joss.01157_

A close description of the vanilla DLM model implemented here can be found in [Laine et al 2014](https://www.atmos-chem-phys.net/14/9707/2014/acp-14-9707-2014.pdf), and this model/code was used for analyzing ozone data in [Ball et al 2017](https://www.research-collection.ethz.ch/handle/20.500.11850/202027) and [Ball et al 2018](https://www.atmos-chem-phys.net/18/1379/2018/acp-18-1379-2018.html). Please consider citing these papers too when you use this code.

## Contributing

Contributions, issues, and feature requests are welcome! 

If you are interested in improving `dlmmc` or fixing a bug, please review our [Contributing Guide](CONTRIBUTING.md) for detailed instructions on how to get started and submit a Pull Request.
