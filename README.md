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
conda create -n dlmmc-cmdstanpy python=3 cmdstanpy xarray netCDF4 -y
conda activate dlmmc-cmdstanpy
# compile the models
python compile_stan_models.py
# Run the benchmark to see if it works
python dlm_benchmark.py

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

This second line compiles all of the DLM models on your machine, saves them in `models/`, and then you're ready to start DLMing! Jump straight into the jupyter notebook tutorial `dlm_tutorial.ipynb` (see below), or if you prefer you can run a test suite to check that the install worked and all models run smoothly by executing (this will take some minutes to run through):
```
jupyter-nbconvert --to notebook --execute --ExecutePreprocessor.timeout=100000 dlm_validation_tests.ipynb
```
Notes:
- CmdStan requires a C++ toolchain to compile models. On macOS install Xcode command-line tools (`xcode-select --install`) or use a conda-provided compiler on CI.
- To keep a macOS laptop awake while compiling/sampling use `caffeinate -i <command>` or run the job on a remote server or in a tmux session.
- The repository includes both workflows; importing the `stan_backend.py` will automatically detect and load either PyStan or cmdstanpy.
- dlmmc v1 assumes pystan version 2, which is incompatible with python >3.8. Newer versions of pystan (3.x), as used here, use a different syntax for the code and sampling.
  
Finally, if you want to see what a successful installation looks like, see [INSTALL.md](INSTALL.md)

### Installation with pip (at your own risk)

Anaconda is not a _requirement_ for installing dlmmc, but is recommended because it works robustly with cmdstanpy/pystan. If you would rather use a different python distribution and `pip3` for installing dependencies, you are welcome to (at your own risk); see the [pystan readthedocs](https://pystan.readthedocs.io/en/latest/installation_beginner.html) for advice on installing pystan using `pip3` if you run into problems. Note that if you do not use Anaconda you will also have to install the other dependencies listed above, ie., 


**Platforms** 

dlmmc has been successfully installed on Mac, Linux and Windows. Note that there are some limitations to the functionality of [pystan on Windows](https://pystan.readthedocs.io/en/latest/windows.html), but these do not restrict the use of the dlmmc package for Windows users.

## Usage

**Functionality (to be updated)**

A detailed annotated tutorial walk-through of how to use the code is given in the jupyter notebook `dlm_tutorial.ipynb` -- this tutorial analyses stratospheric ozone time-series data as a case study. The notebook takes you step-by-step through the complete functionality of the code: loading in your own data, running the DLM model, and processing and plotting the results. The tutorial also serves as a test that the install was successful and the compiled models run smoothly (for a more comprehensive test suite see below).

**Running in parallel with MPI (to be updated)**

It's often necessary to perform regression of a large number time-series (eg., over a grid of observations at different altitudes/latitudes/longitudes) and is advantageous to be able to run these in parallel. Although not a central part of this package, I provide a template example for doing large MPI runs in `dlm_lat_alt_mpi_run.py` - see `MPI-README.md` for a description of getting set up with MPI runs.

**Model descriptions**

Mathematical descriptions of each of the DLM models implemented in this package can be found in the file `models/model_descriptions/model_descriptions.pdf`. This file contains a concise description of the parameters of each model, their physical meanings, and how to refer to them in the code: make sure you have read and understand the model description before running a new model!

**Test suite and code validation (to be updated)**

A more comprehensive test suite is provided in `dlm_validation_tests.ipynb`. In this notebook I run through the suite of DLM models in dlmmc, generating mock data and running the DLM on those mock data, for each model in turn. This acts as both a test suite to check the install has worked robustly (ie., all of the models run to completion without error), and also serves as a set of validation tests demonstrating that the input parameters are recovered correctly, within posterior uncertainties, for each model.

## Citing this code

There is a JOSS paper describing dlmmc v1, which can be found [here](http://joss.theoj.org/papers/10.21105/joss.01157). Please cite this paper if you use or refer to this code, as:

_Alsing, (2019). dlmmc: Dynamical linear model regression for atmospheric time-series analysis. Journal of Open Source Software, 4(37), 1157, https://doi.org/10.21105/joss.01157_

A close description of the vanilla DLM model implemented here can be found in [Laine et al 2014](https://www.atmos-chem-phys.net/14/9707/2014/acp-14-9707-2014.pdf), and this model/code was used for analyzing ozone data in [Ball et al 2017](https://www.research-collection.ethz.ch/handle/20.500.11850/202027) and [Ball et al 2018](https://www.atmos-chem-phys.net/18/1379/2018/acp-18-1379-2018.html). Please consider citing these papers too when you use this code.

## Contributing

Contributions, issues, and feature requests are welcome! 

If you are interested in improving `dlmmc` or fixing a bug, please review our [Contributing Guide](CONTRIBUTING.md) for detailed instructions on how to get started and submit a Pull Request.
