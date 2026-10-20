#!/usr/bin/env python3
"""Scoped existing-user nvcc build, caller owns shared GPU lock."""
import argparse,pathlib,subprocess,shlex
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['native','shared'],required=True);p.add_argument('--output',required=True);p.add_argument('--arch',default='sm_70');a=p.parse_args();B=pathlib.Path(__file__).resolve().parent
cmd=['nvcc','-std=c++17','-O3','-arch='+a.arch,'-Xcompiler','-fPIC','-Xcompiler','-pthread','-x','cu']
if a.mode=='shared':cmd+=['-DPERMUTATE_STUDY_SHARED','-shared']
cmd += [str(B/'full_study.cpp'),str(B/'configured/permutation_testing.cpp'),str(B/'configured/statistical_test.cpp'),str(B/'reference_history/utils.cpp')]+[str(x)for x in sorted((B/'reference_history/bzip').glob('*.cpp'))]+['-o',str(pathlib.Path(a.output).resolve())]
print(shlex.join(cmd),flush=True);subprocess.run(cmd,check=True)
