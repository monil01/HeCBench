#!/usr/bin/env python3
"""Register exact own MPI rank PID before framework import/first CUDA context.
exec preserves PID. External GPU processes are never added to this registry.
"""
import os,sys
from pathlib import Path
rank=int(os.environ['OMPI_COMM_WORLD_RANK']);assert rank in (0,1)
folder=Path(os.environ['PINGPONG_PID_REGISTRY']);assert folder.is_dir()
p=folder/f'rank{rank}.pid';tmp=folder/f'rank{rank}.tmp'
tmp.write_text(str(os.getpid()));tmp.replace(p)
print(f'PINGPONG_PREIMPORT_RANK rank={rank} pid={os.getpid()} executable={sys.argv[1]}',flush=True)
os.execvp(sys.argv[1],sys.argv[1:])
