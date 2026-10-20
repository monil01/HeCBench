#!/usr/bin/env python3
"""Rebuild only the CPU/MPI C ABI; invoke before fresh measurements, never mid-run."""
from pathlib import Path
import subprocess
P=Path(__file__).resolve().parent
subprocess.run(['mpicc','-O3','-fPIC','-shared',str(P/'study_mpi.c'),'-o',str(P/'libstudy_mpi.so')],check=True)
print(P/'libstudy_mpi.so')
