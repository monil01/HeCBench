#!/usr/bin/env python3
from pathlib import Path
import subprocess,json,hashlib,os
P=Path(__file__).resolve().parent;R=P.parent.parent;D=R/'.porting-state/portable_julia/gibbs_build';D.mkdir(parents=True,exist_ok=True)
cmd=[os.environ.get('NVCC','nvcc'),'-O3','-std=c++17','-arch=sm_70','-shared','-Xcompiler=-fPIC',str(P/'study_bridge.cu'),'-o',str(D/'libgibbs_study.so')]
version=subprocess.check_output([cmd[0],'--version'],text=True);subprocess.run(cmd,check=True);print(json.dumps({'command':cmd,'nvcc_version':version,'output':str(D/'libgibbs_study.so'),'sha256':hashlib.sha256((D/'libgibbs_study.so').read_bytes()).hexdigest()},indent=2))
