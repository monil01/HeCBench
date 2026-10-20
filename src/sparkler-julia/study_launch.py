#!/usr/bin/env python3
"""Four source MPI ranks, all co-resident on physical GPU6; central caller owns lock."""
from pathlib import Path
import os,sys
if os.environ.get('CUDA_VISIBLE_DEVICES')!='6':raise SystemExit('Require recorded GPU6-only visibility')
H=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/2025/comm_libs/12.9/hpcx/hpcx-2.22.1')
env=os.environ.copy();env.update(OPAL_PREFIX=str(H/'ompi'),OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_coll='^hcoll',UCX_ERROR_SIGNALS='SIGILL,SIGBUS,SIGFPE',JULIA_CUDA_MEMORY_POOL='none',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env['LD_LIBRARY_PATH']=':'.join(str(H/p)for p in ['ompi/lib','ucx/lib','ucc/lib','hcoll/lib','sharp/lib'])+':/usr/lib/x86_64-linux-gnu:'+env.get('LD_LIBRARY_PATH','')
cmd=[str(H/'ompi/bin/mpirun'),'--bind-to','none','-n','4']+sys.argv[1:]
print('SPARKLER_LAUNCH four ranks share one visible GPU; source has no rank/device assignment',flush=True)
os.execve(cmd[0],cmd,env)
