# GPU DLM Evaluation Runbook

This document describes how to evaluate a GPU implementation of a dlmmc DLM on a workstation or HPC system, and how to compare it with the existing CmdStan CPU and MPI workflows. It is an experiment guide, not a claim that dlmmc currently has a production GPU mode.

## Current Backend Status

- `dlmmc.fit()` and `dlmmc.fit_grid_mpi()` use CmdStanPy. There is no `backend="gpu"` option or GPU-enabled production API. Keep CmdStanPy as the CPU backend and scientific reference.
- The TFP code in this directory is a separate prototype. `tfp_real_grid_reference.py` creates fixed-parameter CmdStan likelihood/gradient references, and `tfp_real_grid_benchmark.py` compares those values on TensorFlow CPU/GPU. This is not a full GPU posterior sampler or a replacement for `fit_grid_mpi()`.
- Past local results are cautionary: the tested TFP sequential Metal path was much slower than CPU for the 32-cell likelihood/gradient workload; GPU sampler attempts were incomplete or failed. PyMC/nutpie used CPU in the tested setup. These results do not predict performance on NVIDIA or AMD hardware.
- A machine having a GPU, a visible device, or a successful matrix multiplication does not prove that the DLM model operations run on the GPU or that sampling is faster.

The goal is portability of the model and evaluation procedure, not one universal binary environment. GPU frameworks, drivers, compiler runtimes, and site modules differ by hardware and machine. Keep host-specific environments separate from production dependencies.

## Choose The Comparison

| Workload | Implementation | Purpose |
| --- | --- | --- |
| One-cell CPU fit | `dlmmc.fit()` / CmdStanPy | Validate the installed package and establish a single-cell CPU reference. |
| Gridded CPU fit | `dlmmc.fit_grid_mpi()` / CmdStanPy | Measure CPU scaling and provide the practical grid baseline. |
| GPU likelihood/gradient | Experimental TFP script | Check numerical agreement and device placement; not a sampling-speed result. |
| GPU posterior sampling | Not implemented as a supported dlmmc backend | Future work after a model-faithful sampler path is selected and validated. |

Stan can use OpenCL for particular supported operations when built and configured accordingly, but the current dlmmc models have not been tested or certified for that route. Treat it as a separate experiment; do not assume enabling an OpenCL build automatically accelerates the sequential Kalman/state-space work.

## Procedure

### 1. Select A Host And Allocation

Use the target machine's current scheduler and accelerator documentation. Do not copy a partition name, QoS, module, GPU request, or launcher command from another machine.

- Ursa is the current documented GPU target: `u1-h100` is the first evaluation candidate; `u1-gh` and `u1-mi300x` are additional exploratory targets. See the [Ursa probe notes](README.md#preparing-an-ursa-gpu-probe) and verify access and current scheduler settings on the live system.
- HERA's public guide has not listed a GPU partition. The user reports GPU nodes, so verify live availability with the scheduler and site support. HERA remains a useful CPU/MPI comparison host either way.
- For a workstation or another center, record the exact GPU model and use that site's supported software stack.

Run the probe inside the allocation, not only on a login node:

```bash
python experiments/backend-evaluation/hpc_environment_probe.py \
  > "$TMPDIR/dlmmc-environment-${SLURM_JOB_ID:-local}.json"
```

Retain the JSON with the run results. Record the repository commit separately. The probe reports installed framework versions, loaded modules when exposed through `LOADEDMODULES`, scheduler fields, visible devices, and tiny framework device operations. It is an inventory check, not a DLM validation. On systems without Slurm, unset fields are expected.

### 2. Create A Host-Specific Environment

1. Follow the center's policy for Python distributions, package channels, compilers, GPU drivers, and environment storage. For RDHPCS, use Miniforge/conda-forge; do not treat either checked-in environment YAML as an Ursa specification.
2. Select a framework/runtime combination supported by both the exact GPU and the site driver/modules. NVIDIA/CUDA, AMD/ROCm, Apple/Metal, and other accelerators need separate compatibility checks. The checked-in `environment-tfp-metal.yml` is macOS/Apple Metal only.
3. Keep the CPU CmdStan environment and the accelerator environment isolated. Pin Python, framework, compiler/runtime, and relevant module versions in the experiment record. Do not add these packages to the production package dependencies.
4. Build/compile on the target architecture where required. For MPI, build `mpi4py` against the same MPI implementation used by the site launcher. Keep compiler caches and per-fit scratch on approved writable filesystems; check quota and purge policy first.

Use `ursa_h100_probe.sbatch.example` only as a starting template. Replace its account, environment activation, and GPU request after verifying them on the live site. The template is intentionally not submission-ready.

### 3. Establish The CPU Controls

Run both controls on the same host class, data, model variant, and intended grid where possible:

1. Run a one-cell CmdStan fit to check compilation and sampling on that host.
2. Run the representative grid with CmdStan MPI. Specify scientific warmup/sample counts explicitly: `dlm_lat_alt_mpi_run.py` defaults to one warmup and two samples, which is only a pipeline smoke test.
3. Measure a single-rank serial run where useful, then increase MPI ranks within the allocation. In `fit_grid_mpi()`, rank 0 initializes and writes the NetCDF file; with more than one rank, the other `N-1` ranks fit cells. Do not count the writer as a fitting worker.
4. Record CPU cores, threads, MPI implementation, rank count, node count, CmdStan compilation/cache state, elapsed time, memory, temporary storage, completed/failed cells, and posterior diagnostics.

Use the site's supported MPI launcher and binding flags on HPC. `mpiexec -n N` is suitable only when it matches the local/site MPI configuration; it is not a universal Slurm launch command.

### 4. Check GPU Placement And Numerical Fidelity

Use the existing fixed-parameter experiment as a first filter, not as evidence of sampler speed:

```bash
# Generate reference data and values with the CmdStanPy environment.
python experiments/backend-evaluation/tfp_real_grid_reference.py \
  --cells 32 --output "$TMPDIR/dlmmc-tfp-real-stan-reference.npz"

# In a compatible TFP environment, test CPU and GPU separately or together.
python experiments/backend-evaluation/tfp_real_grid_benchmark.py \
  --reference "$TMPDIR/dlmmc-tfp-real-stan-reference.npz" \
  --device both --repeats 3
```

If the host has no supported TensorFlow GPU, use `--device cpu`; do not interpret a CPU fallback as a GPU run. A GPU request should fail clearly when placement is unsupported. Confirm the printed likelihood and gradient devices are the intended accelerator. Save likelihood/gradient errors, warm-up/compilation time, steady-state timings, and actual device utilization.

Before implementing or timing full GPU sampling, check model fidelity in this order:

1. Fixed-parameter likelihoods and gradients for the selected DLM against CmdStan, including constrained/unconstrained parameter transforms.
2. Edge cases: missing observations and uncertainties, constant/invalid cells, time-varying uncertainty, regressor alignment, first scored observation, priors, and initialization.
3. Posterior results for representative cells: compare posterior summaries and smoothed-state outputs, and report divergences, R-hat, bulk/tail ESS, acceptance, and retained draws.
4. Representative multi-cell/grid behavior and output handling.

Do not call a path a DLM GPU run based only on framework-device discovery, likelihood timing, or a short sampler smoke test.

### 5. Benchmark End To End

Compare at least these configurations:

- CmdStanPy single-cell CPU and CmdStanPy grid MPI at several rank counts.
- The alternate backend on CPU, when available, to isolate framework/backend effects from accelerator effects.
- The same alternate backend on one GPU, with cells batched only where the implementation preserves independent-cell semantics.
- Multi-GPU or hybrid CPU/GPU work only after one-device correctness and performance are understood.

Use identical input cells, data preprocessing, DLM variant, priors, uncertainty, regressors, output requirements, and sampler quality targets. Record seeds and repeat runs; random-number streams need not match between backends. Report cold-start and warm timings separately, and include the full job wall time for the operational comparison. Split out compilation/JIT, data preparation/transfers, warmup, sampling, summaries, and file writing.

Report both throughput and inference quality:

- End-to-end wall time, cells completed per hour, and per-cell elapsed time.
- ESS/second for key parameters at comparable R-hat, divergences, acceptance, and effective sample counts. A faster run with materially worse mixing is not a speedup.
- CPU/GPU utilization, memory, allocation size, number of fitting workers, failed cells, output size, and temporary-storage use.
- Repeats and variability; do not compare a cold GPU compile with a warmed CPU run without reporting both.

The current evaluation plan has a provisional target of at least 2x end-to-end ESS/second at comparable diagnostics, repeated across runs. This is a proposed decision gate, not an approved requirement or a result. Decide with the project whether that gain justifies the extra backend and maintenance burden.

Avoid oversubscription. Do not launch many MPI ranks that each create GPU workers or target the same GPU unless that strategy has been explicitly measured. Start with one process bound to one GPU and a small batch. For CPU/MPI, sweep ranks and chain parallelism only within allocated cores. Preserve rank 0 as the sole NetCDF writer for the existing grid API.

## Interpretation And Next Decision

Keep CmdStanPy as the production CPU backend. A GPU result should lead to a separate proposal for an optional backend, not a silent change to `dlmmc.fit()`. The proposal should include supported hardware/runtime combinations, model parity, failure behavior, test coverage, installation burden, and a repeatable end-to-end benefit over the best CPU/MPI baseline.

If no candidate passes fidelity, reliability, and performance gates, document the result and continue using CmdStan CPU/MPI. A negative result on one GPU/runtime does not prove that all GPU hardware is slower; equally, a fast microbenchmark on one device does not establish a portable benefit.