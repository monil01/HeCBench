#!/usr/bin/env python3
from pathlib import Path
import argparse,subprocess,json
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['native','shared'],required=True);a.add_argument('--output',required=True);a.add_argument('--arch',default='sm_70');z=a.parse_args();cmd=['/usr/bin/nvcc','-std=c++17','-O3','-arch='+z.arch,'-I'+str(P/'reference_history'),str(P/'full_study.cu'),'-lcrypto','-o',z.output]
if z.mode=='shared':cmd+=['-DSTUDY_SHARED','--shared','-Xcompiler=-fPIC']
print('STUDY_BUILD_COMMAND',json.dumps(cmd),flush=True);subprocess.run(cmd,check=True)
