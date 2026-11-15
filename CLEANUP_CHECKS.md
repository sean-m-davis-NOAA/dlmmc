# Cleanup acceptance record

Candidate: `2.2.0rc2+local`, branch `modernization-local`, checked 2026-11-15 on the Apple Silicon laptop. This is a colleague-testing candidate, not a published release. [Machine-readable evidence](docs/cleanup-acceptance.json) records exact runtime versions, source-module hashes, notebook checksums and remaining gates.

| Check actually run | Result |
|---|---|
| Reconstruct affine from exact pinned base; verify adapter patch and recorded source hashes | PASS |
| Build affine and DLMMC wheels and DLMMC source distribution | PASS |
| Install non-editable wheels in a new isolated Python 3.13 environment; `pip check` | PASS |
| Verify imported package paths and source hashes against installed wheel | PASS |
| CPU/native MPS unit and kernel suite from extracted sdist outside the harness | 461 PASS, 16 native CUDA SKIP, 1 optional Stan test DESELECTED; 84.08 s |
| Final profile/optional-dependency checks after whitespace-only source cleanup | 5 PASS; 1.63 s |
| Pinned affine Torch and complete-state checkpoint adapter tests | 18 PASS; 0.54 s |
| Single-panel tutorial, fresh installed-wheel kernel | PASS; 10.10 s, 3 displayed plots |
| Tensorized parallel tutorial, fresh kernel | PASS; 6.68 s, 2 plots |
| Spatial/ENSO tutorial, fresh kernel | PASS; 2.84 s, 17 plots |
| Validation notebook, fresh kernel | PASS; 2.08 s, 11 plots |
| Visual review of every displayed plot | 33 inspected; cached scientific output/provenance preserved |
| Current documentation links, staged whitespace and credential-pattern/file-size audit | PASS |

All tutorial code/outputs come from the fresh execution; one markdown-only clarification was added afterward to label historical stage04 timings explicitly. No long science fit, full-grid benchmark, Stan replacement reference or calibration campaign was launched for cleanup. The test warnings were JUnit `record_property` format notices and one Numba contiguous-array performance advisory, not failed numerical checks.

The clean environment used Python 3.13, Torch 2.15.0, Numba 0.68.1, NumPy 2.5.4 and affine 1.0.0+dlmmc1. Native MPS was explicitly enabled with fallback/fast math disabled. CUDA skips remain unqualified. Optional Stan integration, Linux CI and external hardware are not claimed as newly passed. The scientific qualifications and unresolved flags in [VALIDATION_STATUS.md](VALIDATION_STATUS.md) are unchanged.

Source distributions include bounded tutorials, test fixtures and their small evidence bundles. Large sampling archives, local environments and redundant notebook history are excluded from the branch; historical evidence remains in the modernization harness. The affine dependency is distributed as a reproducible patch plus license/provenance, with no separate dependency branch push.

After the full suite, one redundant blank line at the end of `profiles.py` was removed; its parsed Python AST is identical. The rebuilt wheel passed the five directly affected profile/optional-dependency checks. All 75 final source modules match the installed wheel. Checksummed CSV line endings and required unified-patch context whitespace are preserved with explicit Git attributes.
