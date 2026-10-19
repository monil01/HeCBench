#!/usr/bin/env python3
"""Owned two-rank launcher. Caller must hold the central GPU6 study lock."""
import os,sys,subprocess
from pathlib import Path
H=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/2025/comm_libs/12.9/hpcx/hpcx-2.22.1')
if os.environ.get('CUDA_VISIBLE_DEVICES')!='5,6':raise SystemExit('Only authorized physicalGPU visibility5,6 accepted')
q=subprocess.run(['nvidia-smi','-i','5','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
if q.stdout.strip():raise SystemExit('GPU5 external compute PID present; defer: '+q.stdout.strip())
print('ALLREDUCE_GPU5_IDLE_GUARD beforelaunch=pass',flush=True)
env=os.environ.copy();paths=[H/x for x in['ompi/lib','ucx/lib','ucc/lib','hcoll/lib','sharp/lib']];env['LD_LIBRARY_PATH']=':'.join(str(x)for x in paths if x.exists())+':'+env.get('LD_LIBRARY_PATH','');env['OPAL_PREFIX']=str(H/'ompi');env['UCX_TLS']='sm,self,cuda_copy,cuda_ipc,tcp';env['UCX_MEMTYPE_CACHE']='n';env['OMPI_MCA_coll']='^hcoll'
if len(sys.argv)<2:raise SystemExit('Provide rank executable/arguments')
cmd=[str(H/'ompi/bin/mpirun'),'-n','2','--mca','pml','ucx']+sys.argv[1:]
os.execve(cmd[0],cmd,env)
