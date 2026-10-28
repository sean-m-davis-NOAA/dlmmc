[![GitHub release](https://img.shields.io/github/v/release/sean-m-davis-NOAA/dlmmc)](https://github.com/sean-m-davis-NOAA/dlmmc/releases/latest)
# DLMMC

### Note: This Python 3 package uses CmdStanPy for Stan model fitting.

**Dynamical Linear Modelling** (DLM) regression code in python for analysis of time-series data. The code is targeted at atmospheric time-series analysis, with a detailed worked example (and data) included for stratospheric ozone, but is a fairly general suite of state space model that can be applied or extended to a wide range of problems.

The core of this package is a suite of DLM models implemented in [stan](https://mc-stan.org), using a combination of HMC sampling and Kalman filtering to infer the DLM model parameters (trend, seasonal cycle, auto-regressive processes etc) given some time-series data. To make the code as accessible as possible, I provide a step-by-step tutorial in python for how to read in your data, run the DLM model(s), and process the outputs to make nice plots. Once you've worked through this tutorial you should have all the tools you need to apply DLM to your own data!

Note: some basic working knowledge of python and jupyter notebooks is required to make the most of this package (but it's not too onerous).

## Installation

### Recommended: install from PyPI with Conda

This setup uses Conda for CmdStanPy and its CmdStan dependency, then installs
the released dlmmc package from PyPI with pip. Notebook and MPI packages are
optional and installed separately below.

#### 1. Check for working C++ compiler, and install if necessary

CmdStan needs a C++ compiler and Make. On macOS or Linux, check in a terminal:

```bash
c++ --version
make --version
```

Both commands should print version information (for example, Apple Clang or GCC,
and GNU Make). If either command is missing:

- **macOS:** run `xcode-select --install`, complete the installer, then repeat
  the checks.
- **Linux or HPC:** follow the
  [compiler setup instructions](INSTALL.md#c-compiler-and-make).

These checks confirm the tools are available; the benchmark below verifies that
CmdStan can compile and run a model.

#### 2. Create conda environment and pip install `dlmmc`

```bash
conda create -n dlmmc-env -c conda-forge python pip cmdstanpy -y
conda activate dlmmc-env
python -m pip install dlmmc
```

Conda installs CmdStan automatically as a dependency of CmdStanPy, so there is
no separate `cmdstan` or `install_cmdstan()` step. No specific Python version
needs to be selected; dlmmc requires Python 3.10 or newer.

**Alternative step 2: Conda-only installation (coming soon).**
The conda-forge recipe is under review. Once it is published, you will be able
to install dlmmc without pip:

```bash
conda create -n dlmmc-env -c conda-forge dlmmc -y
conda activate dlmmc-env
```

Conda will install Python, CmdStanPy, CmdStan, and the other required dependencies
automatically. Until the recipe is available, use the PyPI setup above.

#### 3. Optional: install notebook and example dependencies

After either step 2, install these packages to run the example notebooks,
benchmark, and plots:

```bash
conda install -c conda-forge jupyterlab ipykernel matplotlib scipy netcdf4 "arviz<1" -y
```

The notebooks use the pre-1.0 ArviZ API, so keep the `arviz<1` constraint.

In VS Code, select `dlmmc-env` as the notebook kernel. JupyterLab is not needed
when using VS Code; omit `jupyterlab` from the command if you prefer. Otherwise,
run `jupyter lab` from the activated environment.

#### 4. Optional: install MPI support

For MPI grid fitting on a local macOS/Linux workstation:

```bash
conda install -c conda-forge mpi4py openmpi netcdf4 scipy -y
```

This includes NetCDF support and SciPy for the grid example, even if you skipped
step 3. On HPC systems, use the site's MPI implementation instead of this
command; see [MPI-README.md](MPI-README.md).

#### 5. Optional check: verify imports and CmdStan discovery

```bash
# Expected output: CmdStan: <path to CmdStan in the active Conda environment>
python -c "import cmdstanpy, dlmmc; print('CmdStan:', cmdstanpy.cmdstan_path())"
```

### Development installations

Only contributors need an editable installation: from a cloned repository, run
`python -m pip install -e '.[diagnostics]'`. This makes changes to the local
Python source take effect without reinstalling. The non-editable PyPI install
above is recommended for normal use.

## Getting Started

The example notebooks, scripts, and data are in the repository. To obtain them
after installing dlmmc, clone the repository (or download and extract its source
archive):

```bash
git clone https://github.com/sean-m-davis-NOAA/dlmmc.git
cd dlmmc
```

Cloning provides the examples; it does not require reinstalling the package.
Install the optional notebook/example dependencies in step 3 above before
running the benchmark or notebooks, and MPI support in step 4 for grid fitting.
Run the following from the repository root with `dlmmc-env` activated.

1. **Run the benchmark.** This is the quickest end-to-end check of data loading, model compilation, and CmdStanPy sampling.

```bash
python dlm_benchmark.py
```

The benchmark writes a timestamped JSON file under `results/`. Confirm it contains `compile_seconds`, `fit_seconds`, and posterior dimensions.

Models compile automatically on the first fit. Optionally run
`python compile_stan_models.py` to compile all models in advance.

2. **Work through the tutorial.** Open `dlm_tutorial.ipynb` in JupyterLab or VS Code, select `dlmmc-env` as the kernel, and run the cells in order. The tutorial uses four chains with 1,000 warmup and 2,000 sampling iterations per chain, so it takes longer than the benchmark.

3. **Run the validation suite.** Open `dlm_validation_tests.ipynb` and run the cells in order. It performs synthetic parameter-recovery tests for four model variants using short chains. Review the recovery intervals, R-hat/ESS values, and divergences; the short settings demonstrate the workflow and are not a guarantee of convergence.

4. **Optionally test the gridded MPI workflow.** This fits the example pressure/latitude grid in parallel and streams results to one NetCDF writer. It requires the optional MPI dependencies installed above; on a local Open MPI setup:

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

### Temporary storage

Each fit removes its temporary Stan output after loading the posterior and
diagnostics into memory. Failed or interrupted fits retain their output and log
its location for troubleshooting. Pass `output_dir="/path/to/output"` to
`dlmmc.fit()` to keep raw output; dlmmc never removes user-specified output
directories. The compiled-model cache is separate and is not removed.

A single fit can still produce large CSV files. If temporary storage is limited,
set `TMPDIR` to a writable scratch directory with sufficient space and quota
before starting Python or the notebook kernel. See [INSTALL.md](INSTALL.md).

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
