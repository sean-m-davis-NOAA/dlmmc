# Experimental Backend Evaluation Instructions

## Scope and Goal

These instructions apply to files in `experiments/backend-evaluation/`. The goal is to evaluate alternative execution backends for the existing dlmmc DLMs, with particular interest in GPU acceleration and HPC grid workloads. This is research code, not a production backend.

Keep CmdStanPy as the package default. Do not change `src/dlmmc/`, package dependencies, or the public API from this experiment scope without explicit approval. Keep generated chains, CSVs, model executables, and benchmark outputs outside version control unless a small artifact is intentionally needed to reproduce a result.

For user-facing, host-neutral instructions on preparing and comparing GPU workloads, maintain [GPU_RUNBOOK.md](GPU_RUNBOOK.md). Keep it explicit that the current production API is CmdStanPy-only and that the checked-in GPU work is experimental.

## Model-Fidelity Requirements

Use the packaged Stan implementation as the reference. Before comparing performance, preserve and test:

- Data centering and range scaling, including the same uncertainty scaling.
- Priors and parameter constraints on the same natural scale.
- Initial-state covariance and the convention that the prior-time state transitions before the first scored observation.
- Regressor alignment: the first scored observation uses the first supplied regressor after the unobserved prior-time state.
- Known time-varying observation uncertainty and missing-observation handling.
- Seasonal harmonics, AR state, static/dynamic regression behavior, and returned state summaries for the selected model.

Start with fixed-parameter likelihood and gradient comparisons. Do not claim posterior parity from a likelihood-only test or a short sampler smoke test. For posterior comparisons, report R-hat, bulk/tail ESS, divergences, acceptance, and the number of retained draws.

## Current Baseline

Results recorded in `README.md` are exploratory:

- A real-cell PyMC/nutpie run used four chains, 500 warmup, and 500 retained draws. It was close to CmdStanPy for one ozone cell, but JAX exposed CPU only on the tested Mac.
- A 32-cell real-data TFP likelihood/gradient batch matched the CmdStanPy diagnostic reference closely. In the tested sequential float32 path, CPU was substantially faster than Apple Metal; the parallel TFP sampler path failed in MPSGraph.
- TFP NUTS required a scratch-only workaround for TensorFlow Metal integer stateless random draws. Short chains were not converged; do not promote that private-API workaround into package code.
- The JAX Metal prototype failed on a basic operation in the tested environment.
- The `dlmmc-pymc-eval` environment exposes JAX CPU only and does not include NumPyro. In `jax_dlm`, JAX 0.8.2 and NumPyro 0.19.0 detect `METAL:0`, but a basic matmul fails with `UNIMPLEMENTED: default_memory_space is not supported`. On CPU, the one-cell NumPyro likelihood/gradient and simulation-smoothed outputs match CmdStan. An eight-cell 4x500/500 NumPyro run took 4m34s, versus 4m22s for the best-sized CmdStan MPI baseline; NumPyro had zero divergences, R-hat <= 1.0011, and min bulk ESS of 1,592 (CmdStan: R-hat <=1.0103, min bulk ESS 691). State posterior-mean RMSE was 2.2-3.8% of CmdStan posterior SD. Full-grid scaling remains open.
- Numba 0.68.1 in `dlmmc-local-optimizations` matches a fixed CmdStan AR(1) likelihood to 1.1e-10 with a 0.413 ms warm call after 2.46 s compile-plus-first-call. Its finite-difference gradient is a parity diagnostic only, not a NUTS backend.
- PyTorch 2.9.1 MPS in `dlmmc-torch-mps` passes basic matmul/autograd, but the eight-cell DLM likelihood/gradient batch is slower on MPS (268 ms, float32) than CPU (82 ms, float64), with larger MPS errors. The latest PyPI wheel tested (2.15.0) collided with the Numba environment's OpenMP runtime, so keep Torch isolated.
- The affine sampler v0.1 smoke ran on CPU for a toy Gaussian target, but its Metal path fails at `tf.math.is_nan`; it lacks built-in seeding and convergence diagnostics. No affine DLM posterior test exists.
- A Numba 0.68.1 + emcee 3.1.6 pilot uses a vectorized stretch-move callback with `prange` over 32 walkers. Two eight-cell 5,000-step MPICH seeds took 103.6-113.1 s including JAX CPU smoothing. A third 10,000-step seed took 194.6 s end to end and produced minimum estimated ESS 4,456-5,214, above the matched CmdStan run's minimum bulk ESS 3,084. Against the 8-slot CmdStan reference, parameter mean/SD errors were at most 2.4%/2.0% of CmdStan SD, state mean errors averaged 1.7-1.9% with 2.5% worst cell, and state SD errors averaged 1.0-1.7% with 2.4% worst cell. The 10,000-step estimated aggregate ESS/sec was 3.4-4.7x higher, but emcee still has no independent-chain R-hat and relies on ensemble autocorrelation estimates. Validate with multiple independent ensembles and per-cell diagnostics before increasing grid size.
- Two matched 32-cell, eight-slot Numba/emcee seeds took 778-787 s including smoothing versus 2,633.9 s for CmdStan. Minimum emcee ESS was 4,214-5,369; CmdStan min bulk ESS was 3,106 with max R-hat 1.0024. Parameter mean/SD errors were at most 2.4%/2.1% of CmdStan SD; state mean/SD errors were at most 3.9%/3.7%. Estimated aggregate ESS/sec was 3.0-4.0x higher across both seeds. Emcee's finite-ensemble autocorrelation ESS is not independent-chain R-hat; review cell-level tail diagnostics before larger-grid work.
- A diverse 64-cell, eight-slot Numba/emcee test (48 spatially distributed cells plus 16 high-noise cells) took 1,595-1,627 s including smoothing; matched CmdStan took 5,166.7 s. Both emcee seeds had mean acceptance 0.560 and minimum ESS 4,285-5,283; CmdStan completed all cells with max R-hat 1.0026 and min bulk ESS 3,033. Parameter mean/SD errors were at most 1.9%/1.9% of CmdStan SD; state mean/SD errors were at most 4.2%/4.7%. Estimated aggregate ESS/sec was 2.9-3.7x higher. This is experimental CPU evidence; ensemble autocorrelation ESS is not independent-chain R-hat, and CmdStan still reports non-fatal NaN-scale warnings.
- A 96-cell follow-up completed in CmdStan (7,713 s; max R-hat 1.0038; min bulk ESS 2,596) and two emcee seeds (2,298-2,415 s including smoothing). Despite close aggregate summaries, seed 1's walkers split between modes for cell 160 `sigma_seas`; its seasonal-state mean/SD errors there were 14%/48% of CmdStan SD. Seed 2 matched cell 160 but had state-SD outliers at cell 125 (23.5% trend, 18.5% AR). The finite-ensemble ESS missed this per-cell mixing failure. Do not extrapolate aggregate ESS/sec or proceed to full-grid emcee; investigate cells 160 and 125 with independent ensembles and per-walker traces.
- A targeted 128-walker rerun on cells 125 and 160 with two seeds removed the cell 160 mode split; dense smoothing (~80k draws/cell) reduced state mean/SD errors to <=1.4%/3.6%. A separate rank-normalized split-R-hat using walker 0 from each of four independent ensembles gave max R-hat 1.00993 (cell 160 `rhoAR1`) and min bulk ESS 593-767. This is a limited independent-replicate diagnostic, not a standard ensemble-wide convergence guarantee; retain the 96-cell no-go until difficult cells pass at full settings.
- A full 96-cell rerun with 128 walkers (2,000 warmup/10,000 retained) completed in 10,542.8 s including smoother. Minimum estimated ESS was 17,600-22,500; parameter mean/SD errors were <=1.5%/1.35% of CmdStan SD and state mean/SD errors <=2.68%/2.59%. This clears the earlier cell-160 parity issue for one seed, but an independent full-96 seed and per-cell convergence review are still needed.
- A targeted two-cell follow-up used 128 walkers on cells 125 and 160 for two seeds. The cell 160 `sigma_seas` walker split did not recur, parameter summaries matched CmdStan, and dense smoothing (~80,000 draws/cell) reduced state mean/SD discrepancies to <=1.4%/3.6% of CmdStan SD. This suggests walker count and smoother Monte Carlo precision explain the earlier outlier, but does not validate the full 96-cell run at 128 walkers. Keep per-cell diagnostics and test difficult cells under any proposed production settings.
- The separate `justinalsing/affine` v0.1 TensorFlow package runs a toy CPU target but fails on Metal at `tf.math.is_nan`; it lacks seed and convergence-diagnostic APIs. Prefer emcee for Numba integration; neither is a production backend yet.

These results are not a recommendation to change the production backend. Re-run them after code, dependency, driver, or hardware changes.

## Evaluation Phases

1. **Reproduce and record the baseline.** Record the git commit, OS, Python and package versions, CmdStan version, device list, GPU model, driver/runtime, CPU/memory, scheduler allocation, filesystem, and `TMPDIR`. Separate compilation/JIT time from steady-state evaluation and sampling time.
2. **Harden grid correctness tests.** Add tests for `fit_grid_mpi()` covering serial execution, rank roles, single-writer NetCDF behavior, cell assignment, skipped and failed cells, coordinates, fit statuses, overwrite behavior, and output dimensions. Use small deterministic fixtures; add a real MPI smoke test only where MPI is available.
3. **Establish CPU/MPI performance.** Benchmark representative valid, missing, constant, and difficult cells with fixed model settings and seeds. Sweep total MPI ranks and `parallel_chains` separately and only within allocated resources. For `N > 1`, rank 0 is the sole NetCDF writer and the other `N-1` ranks fit cells; each worker fits assigned cells sequentially. Record physical/logical CPUs and memory. Report wall time, cells/sec, compile/setup overhead, rank utilization, memory, temporary storage, failure counts, and convergence diagnostics. Current Stan models do not use `reduce_sum`/`map_rect`; do not recommend Stan thread settings as if they were enabled. Preserve the rank-0 writer design unless evidence supports a change.
4. **Validate alternate-backend numerics.** Compare all relevant model variants and cases against CmdStanPy: fixed-parameter log likelihoods and gradients first, then posterior summaries and smoothed-state outputs. Include time-varying uncertainty, missing values, multiple regressors, initialization, dtype precision, and constrained-to-unconstrained gradient transforms.
5. **Test accelerator placement and batching.** Confirm actual operation placement rather than relying only on device visibility. Test a single cell, a batch of cells, and the target grid shape. Record compile/JIT, transfer, warm-up, sampling, and post-processing separately. Test CPU fallback explicitly.
   - Add an isolated JAX + NumPyro feasibility path for the local Mac. `jax_dlm` Metal still fails a basic operation. The CPU DLM likelihood/gradient, hyperparameter posterior, and public simulation-smoothed outputs now pass the one-cell comparisons. Keep grid benchmarking open; do not infer GPU support from this CPU result.
6. **Benchmark end-to-end GPU and hybrid grid workloads.** Compare scientifically meaningful fits at equivalent posterior quality, not just likelihood-call throughput. Evaluate one process per GPU, worker/GPU binding, batched cell execution, and CPU MPI versus GPU task assignment. Avoid oversubscription across MPI ranks, chains, BLAS threads, and GPU workers.
7. **Make a go/no-go decision.** A provisional target is at least 2x end-to-end ESS/sec at comparable posterior diagnostics, repeated across runs. Treat this as a proposed threshold until the user approves it. Keep CmdStanPy as the default unless parity, reliability, and a material end-to-end benefit are demonstrated.

## HERA and Other HPC Systems

The public RDHPCS documentation is available at https://docs.rdhpcs.noaa.gov/. Start with the [Ursa User Guide](https://docs.rdhpcs.noaa.gov/systems/ursa_user_guide.html), [Hera User Guide](https://docs.rdhpcs.noaa.gov/systems/hera_user_guide.html), [Python guidance](https://docs.rdhpcs.noaa.gov/software/python/index.html), [module guidance](https://docs.rdhpcs.noaa.gov/software/modules/index.html), [Slurm guidance](https://docs.rdhpcs.noaa.gov/slurm/overview.html), [storage guidance](https://docs.rdhpcs.noaa.gov/data/storage.html), [Jupyter guidance](https://docs.rdhpcs.noaa.gov/software/python/jupyter.html), and [VS Code guidance](https://docs.rdhpcs.noaa.gov/software/VSCode/index.html).

The published Ursa User Guide lists three GPU partitions:

- `u1-h100`: 58 nodes with two NVIDIA H100 GPUs per node, 94 GB per GPU. With a GPU allocation use the `gpu` QoS; CPU-allocation projects may request windfall access using `gpuwf`.
- `u1-gh`: 8 exploratory nodes with one NVIDIA GH200 Grace Hopper superchip per node, available via `gpuwf`.
- `u1-mi300x`: 5 exploratory nodes with eight AMD MI300X APUs per node, available via `gpuwf`.

Begin with `u1-h100` and one GPU for framework/device discovery. The exploratory partitions are alternatives if H100 is unavailable or if hardware diversity is part of the test. The guide lists Intel MPI, HPC-X, and OpenMPI, but the currently tested `mpi4py`/CmdStan stack must be checked against the selected module combination. The published Hera guide lists CPU partitions but no GPU partition; do not assume HERA is a GPU target until current resources are verified there.

RDHPCS Python policy disallows Anaconda/Miniconda distributions; use Miniforge and conda-forge. Pin site module versions and load them inside batch scripts, not shell startup files. Install environments and VS Code server files in project home; run compute-intensive jobs from the site-approved Work/scratch filesystem. Ursa `/scratch[3,4,5]` is not backed up; confirm purge/quota policy before storing output. The VS Code guide describes Remote-SSH through an assigned port tunnel and recommends project-space server storage. For Jupyter, follow the documented tunnels and run the kernel in an interactive compute allocation rather than on a login node.

`environment-tfp-metal.yml` is only for macOS/Apple Metal. Do not use it on Ursa. No Ursa GPU framework environment is pinned yet: first inspect modules and the actual GPU driver/runtime, then choose a compatible TensorFlow/JAX/PyTorch stack and validate it with the environment probe. The checked-in evaluation environments are local snapshots, not certified Ursa lockfiles.

For HERA and other sites, document a site-neutral procedure first, then add verified site-specific examples. In particular:

- Build/install `mpi4py` against the same MPI implementation used by the site's launcher.
- Compile Stan on the target architecture or use a compatible shared cache; test shared-filesystem locking and permissions.
- Put per-fit CmdStan output and temporary compiler/JIT files on approved writable scratch via `TMPDIR`.
- Define scheduler allocation, rank count, tasks per node, GPU binding, CPU/thread limits, and memory requests explicitly.
- Verify only rank 0 writes the final NetCDF file and workers do not concurrently write it.
- Provide a VS Code Remote SSH/dev-container path only after verifying the site's supported access method and authentication policy. Never place passwords, tokens, private keys, or login instructions in the repository.

## Existing Entry Points

Run commands from the repository root:

- `python -m pytest` runs package tests.
- `python dlm_benchmark.py` runs the existing single-cell CmdStanPy benchmark.
- `python dlm_lat_alt_mpi_run.py --help` shows the grid-runner settings; its tiny defaults are a smoke test, not scientific sampling.
- `python experiments/backend-evaluation/cmdstan_real_cell.py` and `pymc_real_cell.py` reproduce the one-cell comparison.
- `python experiments/backend-evaluation/tfp_real_grid_reference.py` and `tfp_real_grid_benchmark.py` reproduce fixed-parameter real-grid comparisons.
- `python experiments/backend-evaluation/hpc_environment_probe.py` emits a JSON hardware/software/device report without installing anything; pass `--output path.json` when the framework may log to stdout.
- The JAX probe reports package availability and device placement, including PyTorch MPS and Numba package status; visibility alone is not evidence that a DLM runs on a GPU.
- `python experiments/backend-evaluation/numpyro_real_cell.py --help` describes the isolated CPU comparison. Run `cmdstan_real_cell.py` first to create its posterior, smoothed states, and fixed likelihood/gradient reference; NumPyro refuses to sample unless fixed-likelihood/gradient parity passes.
- `cmdstan_real_grid_fit.py` writes matched CmdStan posterior references for selected cells; `numpyro_real_grid.py` gates batched NumPyro sampling on fixed likelihood/gradient parity and compares posterior/smoother summaries.
- `conda run -n dlmmc-local-optimizations python experiments/backend-evaluation/numba_real_cell.py --help` describes the CPU-only forward-likelihood JIT benchmark; it is not an end-to-end sampler test.
- `conda run -n dlmmc-local-optimizations python experiments/backend-evaluation/numba_emcee_real_cell.py --help` describes the Numba-vectorized emcee sampler; the separate smoother-check scripts use JAX CPU only for post-processing.
- `experiments/backend-evaluation/numba_emcee_grid_mpi.py` runs independent cell ensembles through MPI; `numba_emcee_grid_smoother_check.py` compares their smoothed state means with a matching CmdStan NetCDF reference.
- `experiments/backend-evaluation/numba_emcee_grid_compare.py` compares two saved emcee runs against CmdStan parameter quantiles and smoothed-state means/SDs.
- `experiments/backend-evaluation/numba_emcee_independent_rhat.py` computes ArviZ rank-normalized split-R-hat and bulk ESS from one prespecified walker trajectory per independent emcee run; it deliberately does not treat an ensemble's walkers as independent chains.
- `conda run -n dlmmc-torch-mps python experiments/backend-evaluation/torch_real_cell.py --help` and `torch_real_grid.py --help` describe the fixed single-cell and batched-grid Torch/MPS kernel checks.
- `python experiments/backend-evaluation/mpi_rank_sweep.py --help` describes a bounded, repeatable local MPI scaling tool; its smoke-test defaults do not establish scientific sampler performance.
- `experiments/backend-evaluation/ursa_h100_probe.sbatch.example` is a one-GPU Slurm probe template. Replace the account and site-verified environment activation before submitting; it is not ready to submit unchanged.

The TFP reference uses CmdStanPy `log_prob`, which launches a diagnostic subprocess per cell. Its wall time is not a fair comparison with a persistent TFP graph or a posterior sampler.

## Reporting

Every experiment report should include the code commit, environment/hardware, data/cell selection, model/priors, chain and warmup settings, output locations, compile and runtime timings, per-cell throughput, posterior diagnostics, numerical differences, and known failures. Label smoke tests as smoke tests. Do not commit generated chain outputs or large temporary data.
