[![GitHub release](https://img.shields.io/github/v/release/sean-m-davis-NOAA/dlmmc)](https://github.com/sean-m-davis-NOAA/dlmmc/releases/latest)
# DLMMC

### Note: This Python 3 package uses CmdStanPy for Stan model fitting.

**Dynamical Linear Modelling** (DLM) regression code in python for analysis of time-series data. The code is targeted at atmospheric time-series analysis, with a detailed worked example (and data) included for stratospheric ozone, but is a fairly general suite of state space model that can be applied or extended to a wide range of problems.

The core of this package is a suite of DLM models implemented in [stan](https://mc-stan.org), using a combination of HMC sampling and Kalman filtering to infer the DLM model parameters (trend, seasonal cycle, auto-regressive processes etc) given some time-series data. To make the code as accessible as possible, I provide a step-by-step tutorial in python for how to read in your data, run the DLM model(s), and process the outputs to make nice plots. Once you've worked through this tutorial you should have all the tools you need to apply DLM to your own data!

Note: some basic working knowledge of python and jupyter notebooks is required to make the most of this package (but it's not too onerous).

## Installation

### Install from PyPI with Conda (available now)

```bash
conda create -n dlmmc-env -c conda-forge python=3.10 cmdstanpy
conda activate dlmmc-env
python -m pip install dlmmc
```

Conda installs CmdStanPy and its CmdStan runtime; pip installs `dlmmc` and its Python dependencies. For optional features, install `python -m pip install 'dlmmc[diagnostics]'` for ArviZ diagnostics or `python -m pip install 'dlmmc[mpi]'` for MPI/NetCDF support. On HPC systems, install `mpi4py` against the site's MPI implementation rather than mixing MPI libraries.

To run the example notebooks, install Jupyter and plotting dependencies with `conda install -c conda-forge jupyterlab ipykernel matplotlib scipy`, then install ArviZ with `python -m pip install 'dlmmc[diagnostics]'`.

CmdStan compiles native Stan executables. A C++ toolchain is required; on macOS, install the Xcode command-line tools (`xcode-select --install`).

### Install directly from conda-forge (not available yet)

The conda-forge recipe is still under review. This simple install will not work until the recipe is merged and the package is published:

```bash
conda create -n dlmmc-env -c conda-forge dlmmc
conda activate dlmmc-env
```

For source development, clone the repository and use `python -m pip install -e .`. See [INSTALL.md](INSTALL.md) for environment and HPC details.

## Getting Started

To run the repository's benchmark and notebooks, clone the source checkout and activate the environment installed above:

```bash
git clone https://github.com/sean-m-davis-NOAA/dlmmc.git
cd dlmmc
```

Run the following commands from the repository root.

1. **Optionally precompile the packaged models.** Models compile lazily on the first `fit()` call; this command precompiles all models up front.

```bash
python compile_stan_models.py
```

2. **Run the benchmark.** This is the quickest end-to-end check of data loading, model compilation, and CmdStanPy sampling.

```bash
python dlm_benchmark.py
```

The benchmark writes a timestamped JSON file under `results/`. Confirm it contains `compile_seconds`, `fit_seconds`, and posterior dimensions.

3. **Work through the tutorial.** Open `dlm_tutorial.ipynb` in JupyterLab or VS Code, select the CmdStanPy environment as the kernel, and run the cells in order. The tutorial uses four chains with 1,000 warmup and 2,000 sampling iterations per chain, so it takes longer than the benchmark.

4. **Run the validation suite.** Open `dlm_validation_tests.ipynb` and run the cells in order. It performs synthetic parameter-recovery tests for four model variants using short chains. Review the recovery intervals, R-hat/ESS values, and divergences; the short settings demonstrate the workflow and are not a guarantee of convergence.

5. **Optionally test the gridded MPI workflow.** This fits the example pressure/latitude grid in parallel and streams results to one NetCDF writer. It requires a working MPI implementation and the `mpi` extra; on a local Open MPI setup:

```bash
mpirun -np 4 python dlm_lat_alt_mpi_run.py
```

The MPI example defaults to very short chains for a pipeline smoke test, not scientific inference. See [MPI-README.md](MPI-README.md) for output verification and HPC launch details.

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

Mathematical descriptions of each DLM model are available in [`model_descriptions/model_descriptions.pdf`](model_descriptions/model_descriptions.pdf). This document describes the model parameters and their physical meanings.

## Validation

`dlm_validation_tests.ipynb` provides synthetic parameter-recovery demonstrations for four CmdStanPy model variants. Interpret its recovery intervals and sampler diagnostics as evidence for each run, not as a guarantee that every parameter is well identified under all settings.

## Citing this code

There is a JOSS paper describing dlmmc v1, which can be found [here](http://joss.theoj.org/papers/10.21105/joss.01157). Please cite this paper if you use or refer to this code, as:

_Alsing, (2019). dlmmc: Dynamical linear model regression for atmospheric time-series analysis. Journal of Open Source Software, 4(37), 1157, https://doi.org/10.21105/joss.01157_

A close description of the vanilla DLM model implemented here can be found in [Laine et al 2014](https://www.atmos-chem-phys.net/14/9707/2014/acp-14-9707-2014.pdf), and this model/code was used for analyzing ozone data in [Ball et al 2017](https://www.research-collection.ethz.ch/handle/20.500.11850/202027) and [Ball et al 2018](https://www.atmos-chem-phys.net/18/1379/2018/acp-18-1379-2018.html). Please consider citing these papers too when you use this code.

## Contributing

Contributions, issues, and feature requests are welcome! 

If you are interested in improving `dlmmc` or fixing a bug, please review our [Contributing Guide](CONTRIBUTING.md) for detailed instructions on how to get started and submit a Pull Request.
