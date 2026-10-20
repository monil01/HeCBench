#!/usr/bin/env python3
"""Central verifier owns GPU6 lock; this entrypoint does not nest a lock."""
import argparse,pathlib,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--models',choices=['native','cuda_jl','ka','jacc','all'],default='all');a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;R=B.parent.parent;C=R/'.porting-state/portable_julia/configs'/('nbnxm_reproduction_'+a.models+'.json');subprocess.run([sys.executable,str(B/'make_config.py'),'--models',a.models,'--output',str(C)],check=True);raise SystemExit(subprocess.run([sys.executable,str(R/'tools/verify_portable_julia.py'),'--config',str(C)],cwd=R).returncode)
