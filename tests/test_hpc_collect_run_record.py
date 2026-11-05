import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "hpc" / "collect_run_record.py"
spec = importlib.util.spec_from_file_location("collect_run_record", SCRIPT)
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)

METADATA = """job_id=42
git_commit=abc123
git_status:
 M dlm_benchmark.py
?? new.py
nodes=2
ntasks=9
cpus_per_task=1
cell_indices=0 22 43
module_list:

Currently Loaded Modules:
  1) gnu/14.2.0   3) impi/2024.2.1
  2) intel/2024.2.1

srun_mpi_plugins:
MPI plugin types are...
\tpmi2
python 3.12.16 (main) [GCC 15.3.0]
cmdstanpy 1.3.2
cmdstan_version (2, 40)
mpi4py 4.1.2
mpi_library Intel(R) MPI Library 2021.13 for Linux* OS
"""


def test_parse_metadata_blocks_and_versions():
    meta = collect.parse_metadata(METADATA)

    assert meta["git_commit"] == "abc123"
    assert meta["git_dirty"] is True
    assert meta["ntasks"] == "9"
    assert meta["cell_indices"] == "0 22 43"
    assert meta["loaded_modules"] == "gnu/14.2.0 intel/2024.2.1 impi/2024.2.1"
    assert meta["cmdstanpy"] == "1.3.2"
    assert meta["mpi_library"].startswith("Intel(R) MPI Library 2021.13")


def test_parse_metadata_clean_tree():
    meta = collect.parse_metadata("git_status:\nnodes=1\n")

    assert meta["git_dirty"] is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2048K", 2.0), ("3M", 3.0), ("1.5G", 1536.0), ("", None)],
)
def test_parse_rss_mib(value, expected):
    assert collect.parse_rss_mib(value) == expected


def test_fit_step_is_longest_numbered_step():
    rows = [
        {"JobID": "7", "ElapsedRaw": "100"},
        {"JobID": "7.batch", "ElapsedRaw": "100"},
        {"JobID": "7.0", "ElapsedRaw": "3"},
        {"JobID": "7.1", "ElapsedRaw": "90"},
    ]

    assert collect.fit_step(rows)["JobID"] == "7.1"


def test_worker_cell_counts_follow_static_assignment():
    cells = [0, 22, 43, 65, 86, 108, 129, 151, 172, 194, 215, 237, 258, 280, 301, 323]

    assert collect.worker_cell_counts(cells, 1) == [16]
    assert collect.worker_cell_counts(cells, 3) == [8, 8]
    assert collect.worker_cell_counts(cells, 9) == [2] * 8
