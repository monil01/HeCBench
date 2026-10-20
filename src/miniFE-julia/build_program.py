#!/usr/bin/env python3
"""Compiler adapter; caller/central verifier owns the study GPU lock."""
import argparse,pathlib,subprocess,shlex
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','native','shared'],required=True);p.add_argument('--source');p.add_argument('--output',required=True);p.add_argument('--arch',default='sm_70');a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;H=B/('reference_history'if a.mode=='witness'else'configured');source=B/(a.source or'full_study.cpp')
cmd=['nvcc','-std=c++17','-O3','-arch='+a.arch,'-x','cu','-DMINIFE_SCALAR=double','-DMINIFE_LOCAL_ORDINAL=int','-DMINIFE_GLOBAL_ORDINAL=int','-DMINIFE_RESTRICT=__restrict__','-DMINIFE_CSR_MATRIX','-DMINIFE_INFO=0','-DMINIFE_KERNELS=0','-Xcompiler','-fPIC']
if a.mode=='shared':cmd+=['-DMINIFE_STUDY_SHARED','-shared']
cmd += ['-I'+str(H/d)for d in ['src','utils','fem']]+[str(source),str(H/'utils/BoxPartition.cpp'),str(H/'src/YAML_Doc.cpp'),str(H/'src/YAML_Element.cpp'),str(H/'utils/param_utils.cpp'),str(H/'utils/utils.cpp'),str(H/'utils/mytimer.cpp'),'-o',str(pathlib.Path(a.output).resolve())]
print(shlex.join(cmd),flush=True);subprocess.run(cmd,check=True)
