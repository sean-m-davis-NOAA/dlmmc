# Development update 2.3.0.dev0

Composable affine models now replace the requirement to choose a preset. Existing strings remain compatible; CPU and GPU consume one compiled layout and versioned checkpoint specification. Preset-equivalent MPS numerical outputs, saved sampling draws and trend summaries reproduce exactly, with unchanged measured performance on the frozen five panels. A level-model covariance cancellation case is repaired algebraically. All compute backends are available for all implemented component models, including no-AR, by explicit user decision; float32 stress failures remain recorded. See [COMPONENTS.md](COMPONENTS.md) and [COMPOSITION_VALIDATION.md](COMPOSITION_VALIDATION.md).

No release or branch push is implied by this local update. CUDA and extended research qualifications remain pending.

# Development candidate 2.2.0rc2+local

Prepared for colleague testing on the modernization branch; no public release or full hardware/scientific qualification is claimed. This snapshot supersedes the older local rc1 package, whose archives remain unchanged.

- Tensorized affine fitting, complete checkpoints, public diagnostics and conditional reconstruction, retaining optional CmdStan and legacy MPI interfaces.
- Six preset model families across CPU NumPy/Numba, Torch CPU, native structured MPS and implemented CUDA paths; GPU filtering and trend moments use float32, CPU production uses float64.
- Production/testing defaults: one ensemble, 128 walkers, 128 burn-in, 256 retained; 2,048 integration draws. Separate four-run qualification profile and scope guards remain.
- Named spatial sigma_trend priors and ENSO lag priors, explicit coordinate/data units and fixed predictor coverage. Four-panel pipeline and synthetic lag-recovery evidence; convergence flags retained.
- Four bounded tutorials, checksum-verified historical science bundles, current evidence summary, pinned sampler patch, standalone installation/testing instructions and CPU CI.

Model composition remains preset-based. Arbitrary component/lag composition is deferred. Native CUDA, broader posterior qualification, long-window joint convergence, scalable full-grid field moves/diagnostics and extended calibration remain pending. Historical float32 failures are preserved; no tolerances, priors or likelihood equations were changed during cleanup. See [VALIDATION_STATUS.md](VALIDATION_STATUS.md).

Sampler dispatch now supports explicit Affine or CmdStan selection, with original-demo CmdStan counts (one chain/1,000 warmup/2,000 retained) and current Affine128/128/256 defaults. Common absolute trend summaries and bounded deterministic component outputs support the rebuilt vanilla AR1 tutorial; full-grid sampling stays explicitly disabled by default. See SAMPLERS.md.
