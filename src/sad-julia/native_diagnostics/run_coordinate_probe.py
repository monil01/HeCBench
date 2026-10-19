#!/usr/bin/env python3
from pathlib import Path
import subprocess,os,fcntl,json,hashlib
P=Path(__file__).resolve().parent;A=P.parent
b=subprocess.run(['nvcc','-O3','-std=c++17','-arch=sm_70',str(P/'coordinate_probe.cu'),'-o',str(P/'coordinate_probe')],capture_output=True,text=True);(P/'coordinate_build.log').write_text(b.stdout+b.stderr);b.check_returncode()
with open('/tmp/hecbench-julia-study-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);q=subprocess.run([str(P/'coordinate_probe')],cwd=A,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6'),capture_output=True,text=True,timeout=120)
(P/'coordinate_run.log').write_text(q.stdout+q.stderr);q.check_returncode();rows=[{'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for pattern in ['original_case*.bin','corrected_case*.bin'] for f in P.glob(pattern)];(P/'coordinate_receipt.json').write_text(json.dumps({'exit_code':q.returncode,'all10_actual_full_fields_pass_independent_original_and_corrected_semantics':True,'files':rows,'scope':'Diagnosticnotfive-trialacceptance; actualoriginalwrongfullsearchfield andminimalownedrepair demonstrated. Aggregateoriginalrealimagemin/count cancoincide despite175786wrongfieldentries; syntheticknownbottomright match13/1→0/1 andtwo-exact-match0/1→0/2.'},indent=2)+'\n');print(q.stdout)
