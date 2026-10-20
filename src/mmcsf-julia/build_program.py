#!/usr/bin/env python3
"""Caller owns GPU lock; source GPU bodies unchanged, scoped shared C ABI."""
import pathlib,argparse,subprocess,shlex
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['native','shared'],required=True);p.add_argument('--output',required=True);p.add_argument('--arch',default='sm_70');a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;cmd=['nvcc','-std=c++17','-O3','-arch='+a.arch,'-x','cu','-I/usr/include','-Xcompiler','-fPIC']
if a.mode=='shared':cmd+=['-DMMCSF_STUDY_SHARED','-shared']
cmd += [str(B/'full_study.cpp'),str(B/'configured/kernels.cu'),str(B/'configured/mttkrp_cpu.cu'),'-lpthread','-o',str(pathlib.Path(a.output).resolve())];print(shlex.join(cmd),flush=True);subprocess.run(cmd,check=True)
