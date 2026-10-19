#!/usr/bin/env python3
from pathlib import Path
import subprocess,os,fcntl,json,time
P=Path(__file__).resolve().parent;S=P.parent.parent/'sad-cuda/main.cu';A=P.parent/'original_assets'
b=subprocess.run(['nvcc','-O3','-std=c++17','-arch=sm_70','-I'+str(A),str(S),'-o',str(P/'original_cli')],capture_output=True,text=True);(P/'build_with_DVC_header.log').write_text(b.stdout+b.stderr);b.check_returncode()
with open('/tmp/hecbench-julia-study-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);env=dict(os.environ,CUDA_VISIBLE_DEVICES='6');start=time.time()
 try:
  q=subprocess.run([str(P/'original_cli'),str(A/'coins.bmp'),str(A/'coin.bmp'),'1'],capture_output=True,text=True,env=env,timeout=90);(P/'original_real_assets_R1.log').write_text(q.stdout+q.stderr);out={'exit_code':q.returncode,'wall_seconds':time.time()-start,'original_DVC_assets_used':True,'repeat':1,'scope':'Untouched original source/full original real-image CLI readiness only; independent fullsearch numerical audit pending.'}
 except subprocess.TimeoutExpired as e:
  (P/'original_real_assets_R1.log').write_bytes((e.stdout or b'')+(e.stderr or b''));out={'timeout_seconds':90,'original_DVC_assets_used':True,'scope':'Actual native readiness timeout, no cause assumed.'}
(P/'original_cli_readiness.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
