"""Execute bounded tutorials in fresh kernels without replacing source notebooks."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    names = ("dlm_tutorial.ipynb", "dlm_tensor_grid_tutorial.ipynb",
             "dlm_spatial_priors_tutorial.ipynb", "dlm_validation_tests.ipynb")
    parser.add_argument("--notebook", choices=names, action="append",
                        help="Execute only named notebooks; default is all four")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for key in ("MPLCONFIGDIR", "JUPYTER_RUNTIME_DIR", "IPYTHONDIR", "JUPYTER_CONFIG_DIR"):
        folder = out / key.lower()
        folder.mkdir()
        os.environ[key] = str(folder)
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    records = []
    for name in args.notebook or names:
        source = root / name
        nb = nbformat.read(source, 4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                compact = "".join(cell.source.split())
                if any(flag in compact for flag in ("RUN_SELECTED_FIT=True", "RUN_LAG_FIT=True",
                                                    "RUN_FULL_GRID=True", "RUN_STAN=True")):
                    raise ValueError("Disable spatial and lag long-fit switches before smoke execution")
        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        start = time.perf_counter()
        try:
            NotebookClient(nb, km=km, timeout=180,
                           resources={"metadata": {"path": str(root)}}).execute()
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
        destination = out / name
        nbformat.write(nb, destination)
        plots = {}
        for ci, cell in enumerate(nb.cells):
            for oi, output in enumerate(cell.get("outputs", [])):
                if "image/png" in output.get("data", {}):
                    plot = out / f"{source.stem}-{ci}-{oi}.png"
                    plot.write_bytes(base64.b64decode(output["data"]["image/png"]))
                    plots[plot.name] = hashlib.sha256(plot.read_bytes()).hexdigest()
        records.append({"notebook": name, "status": "EXECUTED_FRESH_KERNEL",
                        "seconds": time.perf_counter()-start, "interpreter": sys.executable,
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "executed_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        "plots": plots, "visual_review": "PENDING"})
        (out / "execution.json").write_text(json.dumps(records, indent=2)+"\n")
        print(name, f"{records[-1]['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
