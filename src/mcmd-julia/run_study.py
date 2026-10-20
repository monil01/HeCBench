#!/usr/bin/env python3
"""Clean unprivileged entrypoint: central verifier alone owns the GPU6 lock."""
import argparse,pathlib,tempfile,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--models',choices=['native','cuda_jl','ka','jacc','all'],default='all');a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;R=B.parent.parent
with tempfile.TemporaryDirectory(prefix='hecbench-mcmd-config-')as d:
 config=pathlib.Path(d)/'config.json'
 subprocess.run([sys.executable,str(B/'make_config.py'),'--models',a.models,'--output',str(config)],check=True,cwd=R)
 subprocess.run([sys.executable,str(R/'tools/verify_portable_julia.py'),'--config',str(config)],check=True,cwd=R)
