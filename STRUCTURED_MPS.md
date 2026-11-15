# Native MPS likelihood and trend smoothing

`fit_production(..., backend="mps")` selects float32 native Metal filtering and default GPU trend smoothing. All six registered model families are specialized by model and regressor count. This is not a general sparse/autograd solver; matrices come from the shared DLMMC model specification. Native MPS requires `torch.mps.compile_shader` (tested with Torch 2.15.0).

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python your_script.py
```

The likelihood runs the complete time loop on the Apple GPU, one thread per walker/panel pair, with every selected panel in each likelihood half-step. Known transition zeros reduce arithmetic; current covariance remains dense. Missing observations skip observation arithmetic. The smoother uses Joseph filtering, diagonally equilibrated partial-pivot backward solves and anchor cross-time covariance, without jitter or floors. Draw tiles bound filter/smoother histories while panels stay together, and GPU-side mixture reductions preserve conditional uncertainty. `workspace_bytes` controls the smoother memory budget. Sampling checkpoints every 16 sweeps do not divide panels into groups of 16.

Per-draw regressors support ENSO interpolation. Four-cell spatial/lag kernel checks and bounded pipeline/recovery runs have executed; short-run convergence remains inconclusive. CPU/native MPS fixed checks cover all six families, including singleton, permutation, chunking and cross-talk. CUDA uses portable Torch equations and remains native-untested.

User-approved production availability does not imply blanket float32 qualification. Saved adverse likelihood/proposal failures and short-chain posterior discrepancies remain. See [VALIDATION_STATUS.md](VALIDATION_STATUS.md) for measured timing, support limits and evidence. Earlier hybrid CPU-smoother runs are historical results; current MPS production includes GPU smoothing.
