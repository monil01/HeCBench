#!/usr/bin/env python3
from pathlib import Path
import subprocess,os,fcntl,json
P=Path(__file__).resolve().parent;A=P.parent/'original_assets'
with open('/tmp/hecbench-julia-study-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);q=subprocess.run(['/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/compute-sanitizer/compute-sanitizer','--tool','synccheck','--error-exitcode','99',str(P/'original_cli'),str(A/'coins.bmp'),str(A/'coin.bmp'),'1'],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6'),timeout=120)
(P/'original_real_assets_synccheck.log').write_text(q.stdout+q.stderr);d={'exit_code':q.returncode,'ERROR_SUMMARY_0':'ERROR SUMMARY: 0 errors' in q.stdout+q.stderr,'source':'Unmodified3kernels and original DVC630x459/116x116inputs; compact177160count haspartialfinalblock.','scope':'Actual named synccheck diagnostic, not proof forall geometries/compiler/environments. No automatic correction or general barrierdefect inference.'};(P/'synccheck_receipt.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
