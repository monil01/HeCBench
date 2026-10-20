#!/usr/bin/env python3
import argparse,pathlib,subprocess
P=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['native','shared'],required=True);p.add_argument('--output',required=True);p.add_argument('--arch',default='sm_70');a=p.parse_args()
c=['/usr/bin/nvcc','-std=c++17','-O3','-arch='+a.arch,str(P/'full_study.cu'),'-lcublas','-o',a.output]
if a.mode=='shared':c+=['-DSTUDY_SHARED','--shared','-Xcompiler=-fPIC']
subprocess.run(c,check=True)
