#!/usr/bin/env python3
"""Central verifier owns GPU6 lock: do not nest a second flock."""
import argparse,pathlib,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--models',choices=['native','cuda_jl','ka','jacc','all'],default='all');a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;R=B.parents[1];C=R/'.porting-state/portable_julia/configs'/('permutate_reproduction_'+a.models+'.json');subprocess.run([sys.executable,str(B/'make_config.py'),'--models',a.models,'--output',str(C)],check=True);raise SystemExit(subprocess.run([sys.executable,str(R/'tools/verify_portable_julia.py'),'--config',str(C)],cwd=R).returncode)
