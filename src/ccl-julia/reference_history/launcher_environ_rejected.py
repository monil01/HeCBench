#!/usr/bin/env python3
"""Two local ranks on authorized GPUs5,6; caller holds central study lock."""
import os,sys,subprocess,uuid,time,signal,json
if os.environ.get('CUDA_VISIBLE_DEVICES')!='5,6':raise SystemExit('Require visibility5,6')
def apps():
 p=subprocess.run(['nvidia-smi','-i','5','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
 return [int(x.strip())for x in p.stdout.splitlines() if x.strip()]
if apps():raise SystemExit('GPU5 occupied; defer')
env=os.environ.copy();nonce=uuid.uuid4().hex;env.update(HECBENCH_CCL_JOB=nonce,NCCL_DEBUG='WARN',OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
p=subprocess.Popen(['/usr/bin/mpirun','--bind-to','none','-n','2']+sys.argv[1:],env=env,start_new_session=True)
checks=[];bad=[]
try:
 while p.poll() is None:
  for pid in apps():
   try:raw=open(f'/proc/{pid}/environ','rb').read()
   except FileNotFoundError:continue
   if ('HECBENCH_CCL_JOB='+nonce).encode() not in raw.split(b'\0'):bad.append(pid)
  if bad:os.killpg(p.pid,signal.SIGTERM);raise RuntimeError('External GPU5 activity; own job aborted: '+str(bad))
  checks.append(time.time());time.sleep(.5)
 rc=p.wait()
finally:
 if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait()
 print('CCL_GPU5_GUARD '+json.dumps({'checks':len(checks),'external_pids':bad,'scope':'Only own process group terminated if external GPU5 use detected'}),flush=True)
sys.exit(rc)
