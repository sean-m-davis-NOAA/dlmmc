[![GitHub release](https://img.shields.io/github/v/release/sean-m-davis-NOAA/dlmmc)](https://github.com/sean-m-davis-NOAA/dlmmc/releases/latest)
# DLMMC

### Note: This package is an updated version of [dlmmc](https://github.com/justinalsing/dlmmc) that uses python3 and cmdstanpy (or optionally, pystan3)

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

The most painless way to get set up is using the [Anaconda python distribution](https://www.anaconda.com/distribution/) (recommended). Two supported installation routes are provided below.

CmdStan produces native compiled Stan executables and cmdstanpy is a lightweight, well-maintained Python interface. This combination is generally faster and more robust than older PyStan workflows. 

**Only choose one installation method from below**

1) CmdStan / CmdStanPy (recommended modern backend)
```
# create and activate a modern environment
conda create -n dlmmc-cmdstanpy -c conda-forge python=3.10 cmdstanpy netCDF4 numpy scipy xarray arviz tqdm mpi4py openmpi matplotlib jupyter ipykernel pandas -y
conda activate dlmmc-cmdstanpy
```

2) PyStan3 (legacy)
```
# create and activate a modern environment
conda create -n dlmmc-pystan python=3 pystan xarray netCDF4 -y
conda activate dlmmc-pystan
# compile the models
python compile_stan_models.py
# Run the benchmark to see if it works
python dlm_benchmark.py
```

Notes:
- CmdStan requires a C++ toolchain to compile models. On macOS install Xcode command-line tools (`xcode-select --install`) or use a conda-provided compiler on CI.
- The CmdStanPy environment command includes the dependencies for the MPI example. On HPC systems, use the site-provided MPI implementation and ensure `mpi4py` is built against it instead of mixing in a separate Conda MPI implementation.
- To keep a macOS laptop awake while compiling/sampling use `caffeinate -i <command>` or run the job on a remote server or in a tmux session.
- The repository includes both workflows; importing the `stan_backend.py` will automatically detect and load either PyStan or cmdstanpy.
- dlmmc v1 assumes pystan version 2, which is incompatible with python >3.8. Newer versions of pystan (3.x), as used here, use a different syntax for the code and sampling.
  
Finally, if you want to see what a successful installation looks like, see [INSTALL.md](INSTALL.md)

## Getting Started

The recommended path below uses the CmdStanPy environment above. Run commands from the repository root.

1. **Install CmdStan.** A C++ toolchain is required; see the installation notes above.

```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

2. **Generate model sources and markers.** This creates the canonical model markers used by the scripts and notebooks; no manual pickle copying is needed.

```bash
python compile_stan_models.py
```

3. **Run the benchmark.** This is the quickest end-to-end check of data loading, model compilation, and CmdStanPy sampling.

```bash
python dlm_benchmark.py
```

The benchmark writes a timestamped JSON file under `results/`. Confirm that its `backends.cmdstanpy` entry includes `compile_time` and `sample_time`, with no `cmdstanpy_error` entry.

4. **Work through the tutorial.** Open `dlm_tutorial.ipynb` in JupyterLab or VS Code, select the CmdStanPy environment as the kernel, and run the cells in order. The tutorial uses four chains with 1,000 warmup and 2,000 sampling iterations per chain, so it takes longer than the benchmark.

5. **Run the validation suite.** Open `dlm_validation_tests.ipynb` and run the cells in order. It performs synthetic parameter-recovery tests for four model variants using short chains. Review the recovery intervals, R-hat/ESS values, and divergences; the short settings demonstrate the workflow and are not a guarantee of convergence.

6. **Optionally test the gridded MPI workflow.** This fits the example pressure/latitude grid in parallel and streams results to one NetCDF writer. It requires a working MPI implementation and `mpi4py`; on a local Open MPI setup:

```bash
mpirun -np 4 python dlm_lat_alt_mpi_run.py
```

The MPI example defaults to very short chains for a pipeline smoke test, not scientific inference. See [MPI-README.md](MPI-README.md) for output verification and HPC launch details.

### Installation with pip (at your own risk)

Anaconda is not a _requirement_ for installing dlmmc, but is recommended because it works robustly with cmdstanpy/pystan. If you would rather use a different python distribution and `pip3` for installing dependencies, you are welcome to (at your own risk); see the [pystan readthedocs](https://pystan.readthedocs.io/en/latest/installation_beginner.html) for advice on installing pystan using `pip3` if you run into problems. Note that if you do not use Anaconda you will also have to install the other dependencies listed above, ie., 


**Platforms** 

dlmmc has been successfully installed on Mac, Linux and Windows. Note that there are some limitations to the functionality of [pystan on Windows](https://pystan.readthedocs.io/en/latest/windows.html), but these do not restrict the use of the dlmmc package for Windows users.

## Usage

Follow the [Getting Started](#getting-started) sequence for the benchmark, tutorial, and validation suite. For large gridded analyses, see `dlm_lat_alt_mpi_run.py` and [MPI-README.md](MPI-README.md).

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
