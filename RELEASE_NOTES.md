# Development candidate 2.2.0rc2+local

Prepared for colleague testing on the modernization branch; no public release or full hardware/scientific qualification is claimed. This snapshot supersedes the older local rc1 package, whose archives remain unchanged.

- Tensorized affine fitting, complete checkpoints, public diagnostics and conditional reconstruction, retaining optional CmdStan and legacy MPI interfaces.
- Six preset model families across CPU NumPy/Numba, Torch CPU, native structured MPS and implemented CUDA paths; GPU filtering and trend moments use float32, CPU production uses float64.
- Production/testing defaults: one ensemble, 128 walkers, 128 burn-in, 256 retained; 2,048 integration draws. Separate four-run qualification profile and scope guards remain.
- Named spatial sigma_trend priors and ENSO lag priors, explicit coordinate/data units and fixed predictor coverage. Four-panel pipeline and synthetic lag-recovery evidence; convergence flags retained.
- Four bounded tutorials, checksum-verified historical science bundles, current evidence summary, pinned sampler patch, standalone installation/testing instructions and CPU CI.

Model composition remains preset-based. Arbitrary component/lag composition is deferred. Native CUDA, broader posterior qualification, long-window joint convergence, scalable full-grid field moves/diagnostics and extended calibration remain pending. Historical float32 failures are preserved; no tolerances, priors or likelihood equations were changed during cleanup. See [VALIDATION_STATUS.md](VALIDATION_STATUS.md).
