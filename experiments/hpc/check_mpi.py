#!/usr/bin/env python3
"""Verify the Slurm-launched MPI communicator before fitting."""

import os

from mpi4py import MPI


comm = MPI.COMM_WORLD
expected = int(os.environ['SLURM_NTASKS'])
total = comm.allreduce(1)
print(f'rank={comm.rank} size={comm.size} sum={total} host={MPI.Get_processor_name()}', flush=True)
if comm.rank == 0:
    print(MPI.Get_library_version(), flush=True)
if comm.size != expected or total != expected:
    raise SystemExit(f'Expected {expected} cooperating MPI ranks.')
