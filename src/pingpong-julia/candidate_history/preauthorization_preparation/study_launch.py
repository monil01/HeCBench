#!/usr/bin/env python3
"""Two local ranks on authorized GPUs5,6; caller holds central study lock."""
import os,sys,subprocess,uuid,time,signal,json,tempfile
from pathlib import Path
if os.environ.get('CUDA_VISIBLE_DEVICES')!='5,6':raise SystemExit('Require visibility5,6')
def apps():
 p=subprocess.run(['nvidia-smi','-i','5','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
 return [int(x.strip())for x in p.stdout.splitlines() if x.strip()]
if apps():raise SystemExit('GPU5 occupied; defer')
H=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/2025/comm_libs/12.9/hpcx/hpcx-2.22.1')
env=os.environ.copy();nonce=uuid.uuid4().hex;env.update(HECBENCH_PINGPONG_JOB=nonce,NPINGPONG_DEBUG='WARN',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env.update(OPAL_PREFIX=str(H/'ompi'),UCX_TLS='sm,self,cuda_copy,cuda_ipc,tcp',UCX_MEMTYPE_CACHE='n',UCX_ERROR_SIGNALS='SIGILL,SIGBUS,SIGFPE',JULIA_CUDA_MEMORY_POOL='none')
env['LD_LIBRARY_PATH']=':'.join(str(H/x)for x in ['ompi/lib','ucx/lib','ucc/lib','hcoll/lib','sharp/lib'])+':/opt/nvidia/hpc_sdk/Linux_x86_64/2025/comm_libs/12.9/nccl/lib:'+env.get('LD_LIBRARY_PATH','')
env['OMPI_MCA_coll']='^hcoll'
registry=tempfile.TemporaryDirectory(prefix='ccl-rank-pids-');env['PINGPONG_PID_REGISTRY']=registry.name
p=subprocess.Popen([str(H/'ompi/bin/mpirun'),'--bind-to','none','-n','2','--mca','pml','ucx']+sys.argv[1:],env=env,start_new_session=True)
checks=[];bad=[]
try:
 while p.poll() is None:
  for pid in apps():
   known={int(f.read_text())for f in Path(registry.name).glob('rank*.pid')}
   if pid not in known:bad.append(pid)
  if bad:os.killpg(p.pid,signal.SIGTERM);raise RuntimeError('External GPU5 activity; own job aborted: '+str(bad))
  checks.append(time.time());time.sleep(.5)
 rc=p.wait()
finally:
 if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait()
 print('PINGPONG_GPU5_GUARD '+json.dumps({'registered_rank_pids':sorted(int(f.read_text())for f in Path(registry.name).glob('rank*.pid')),'checks':len(checks),'external_pids':bad,'scope':'Only own process group terminated if external GPU5 use detected'}),flush=True)
sys.exit(rc)
