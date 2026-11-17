# Install this development branch

Use a fresh virtual environment. The complete laptop test environment uses **Python 3.13** (base package metadata still permits Python 3.10; the current diagnostic extra requires a newer Python). These instructions are run from the DLMMC repository root. They do not change any global Python environment.

```sh
# For a fresh checkout:
git clone --branch modernization-local https://github.com/sean-m-davis-NOAA/dlmmc.git
cd dlmmc

python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip

# Exact reviewed sampler revision; do not replace with pip install affine.
git clone https://github.com/Cosmo-Pop/affine.git .deps/affine
git -C .deps/affine checkout --detach 5fdd3843b099a2386f75d97b6c059f64560afdf9
python -c "import hashlib,json,pathlib; p=pathlib.Path('vendor'); m=json.loads((p/'affine-provenance.json').read_text()); assert hashlib.sha256((p/'affine-modernization.patch').read_bytes()).hexdigest()==m['patch_sha256']"
git -C .deps/affine apply --check ../../vendor/affine-modernization.patch
git -C .deps/affine apply ../../vendor/affine-modernization.patch
python -m pip install './.deps/affine[torch]'
python -m pip install '.[cpu,notebooks,diagnostics,test]'
python -m pip check
```

The patch adds complete-state checkpoint streaming, adapter tests and a local version label (`1.0.0+dlmmc1`); it does not replace the merged upstream sampler algorithms. `vendor/affine-provenance.json` records the base and file hashes; the original affine license is reproduced in `vendor/affine-LICENSE`. A separate affine branch is unnecessary. Keep the pinned revision until a separately reviewed update.

For local code development use `pip install -e '.[cpu,notebooks,diagnostics,test]'` after installing the sampler. The documented notebook bundle lives in the source checkout, not inside the wheel. Source distributions include the tutorials, their evidence bundle and tests.

## Compute backends

Automatic backend selection prefers CUDA, then MPS, then Numba CPU. Numba defaults to float64 and also accepts float32; eight threads are used unless overridden. Torch CPU is retained only as an internal numerical oracle. Native MPS requires Apple Silicon and a Torch build with `torch.mps.compile_shader`; **Torch 2.15.0 was tested**. The broad base Torch dependency does not promise that older builds provide this MPS feature. Launch MPS work with:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python your_script.py
```

For CUDA, install the appropriate CUDA-enabled Torch build before DLMMC. CUDA is implemented but has not been validated on native hardware. Backend unavailability raises an error, and GPU arithmetic defaults to float32; CUDA also accepts explicit float64. MPS cannot use float64. There is no automatic CPU or mixed-precision fallback.

CmdStan is optional: install `'.[stan]'` and a compatible local CmdStan toolchain only when using the legacy fitting API or reference integration test. PyStan is not required. See [TESTING.md](TESTING.md) for bounded checks and current limitations.
