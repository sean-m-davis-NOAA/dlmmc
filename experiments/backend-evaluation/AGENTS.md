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
- The `dlmmc-pymc-eval` environment exposes JAX CPU only and does not include NumPyro. In `jax_dlm`, JAX 0.8.2 and NumPyro 0.19.0 detect `METAL:0`, but a basic matmul fails with `UNIMPLEMENTED: default_memory_space is not supported`. On CPU, the experimental one-cell NumPyro model matches a fixed CmdStan likelihood to 2.3e-10 and maximum relative gradient to 1.35e-12; two 4x500/500 fits had zero divergences and R-hat <= 1.0044. This covers four hyperparameters only; smoothed state summaries and grid performance remain untested.

These results are not a recommendation to change the production backend. Re-run them after code, dependency, driver, or hardware changes.

## Evaluation Phases

1. **Reproduce and record the baseline.** Record the git commit, OS, Python and package versions, CmdStan version, device list, GPU model, driver/runtime, CPU/memory, scheduler allocation, filesystem, and `TMPDIR`. Separate compilation/JIT time from steady-state evaluation and sampling time.
2. **Harden grid correctness tests.** Add tests for `fit_grid_mpi()` covering serial execution, rank roles, single-writer NetCDF behavior, cell assignment, skipped and failed cells, coordinates, fit statuses, overwrite behavior, and output dimensions. Use small deterministic fixtures; add a real MPI smoke test only where MPI is available.
3. **Establish CPU/MPI performance.** Benchmark representative valid, missing, constant, and difficult cells with fixed model settings and seeds. Sweep total MPI ranks and `parallel_chains` separately and only within allocated resources. For `N > 1`, rank 0 is the sole NetCDF writer and the other `N-1` ranks fit cells; each worker fits assigned cells sequentially. Record physical/logical CPUs and memory. Report wall time, cells/sec, compile/setup overhead, rank utilization, memory, temporary storage, failure counts, and convergence diagnostics. Current Stan models do not use `reduce_sum`/`map_rect`; do not recommend Stan thread settings as if they were enabled. Preserve the rank-0 writer design unless evidence supports a change.
4. **Validate alternate-backend numerics.** Compare all relevant model variants and cases against CmdStanPy: fixed-parameter log likelihoods and gradients first, then posterior summaries and smoothed-state outputs. Include time-varying uncertainty, missing values, multiple regressors, initialization, dtype precision, and constrained-to-unconstrained gradient transforms.
5. **Test accelerator placement and batching.** Confirm actual operation placement rather than relying only on device visibility. Test a single cell, a batch of cells, and the target grid shape. Record compile/JIT, transfer, warm-up, sampling, and post-processing separately. Test CPU fallback explicitly.
   - Add an isolated JAX + NumPyro feasibility path for the local Mac. The separate `jax_dlm` environment provides NumPyro; its Metal device fails a basic operation, but the CPU DLM likelihood/gradient and four-hyperparameter posterior pass the current one-cell comparisons. Keep state-smoother parity and grid benchmarking as open gates; do not infer GPU support from this CPU result.
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
- The JAX probe checks package availability and device placement only; it is not evidence that a DLM runs on a GPU. NumPyro is currently absent from the evaluation environment.
- The JAX probe checks package availability and device placement only; it is not evidence that a DLM runs on a GPU. The user's `jax_dlm` environment includes NumPyro; `dlmmc-pymc-eval` does not.
- `python experiments/backend-evaluation/numpyro_real_cell.py --help` describes the isolated CPU comparison. Run `cmdstan_real_cell.py` first to create its posterior and fixed likelihood/gradient reference; NumPyro refuses to sample unless that parity check passes.
- `python experiments/backend-evaluation/mpi_rank_sweep.py --help` describes a bounded, repeatable local MPI scaling tool; its smoke-test defaults do not establish scientific sampler performance.
- `experiments/backend-evaluation/ursa_h100_probe.sbatch.example` is a one-GPU Slurm probe template. Replace the account and site-verified environment activation before submitting; it is not ready to submit unchanged.

The TFP reference uses CmdStanPy `log_prob`, which launches a diagnostic subprocess per cell. Its wall time is not a fair comparison with a persistent TFP graph or a posterior sampler.

## Reporting

Every experiment report should include the code commit, environment/hardware, data/cell selection, model/priors, chain and warmup settings, output locations, compile and runtime timings, per-cell throughput, posterior diagnostics, numerical differences, and known failures. Label smoke tests as smoke tests. Do not commit generated chain outputs or large temporary data.
