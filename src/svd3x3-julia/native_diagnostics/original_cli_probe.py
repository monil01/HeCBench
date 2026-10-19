#!/usr/bin/env python3
from pathlib import Path
import subprocess,os,fcntl,json,time
P=Path(__file__).resolve().parent;A=P.parent/'original_assets/Dataset_1M.txt'
with open('/tmp/hecbench-julia-study-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);start=time.time();q=subprocess.run(['/tmp/hecbench-svd3x3-original',str(A),'1'],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6'),timeout=180)
(P/'original_DVC_R1.log').write_text(q.stdout+q.stderr);(P/'original_CLI_receipt.json').write_text(json.dumps({'exit_code':q.returncode,'wall_seconds':time.time()-start,'repeat':1,'DVC_matrices':1048576,'original_source_and_fast_math_flags':True,'weak_checker':'OriginalCPU/GPUchecker compares onlyfirstn of21n outputs, sharedSVDimplementation, returns0evenFAIL. Readinessnotindependentmathematicalcertificate.','independent_full_U_S_V_check_pending':True},indent=2)+'\n');print(q.stdout);q.check_returncode()
