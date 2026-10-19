#!/usr/bin/env python3
from pathlib import Path
import subprocess,fcntl,os,json,hashlib
P=Path(__file__).resolve().parent;b=subprocess.run(['nvcc','-O3','-use_fast_math','-std=c++17','-arch=sm_70',str(P/'full_dataset_probe.cu'),'-o',str(P/'full_dataset_probe')],capture_output=True,text=True);(P/'full_probe_build.log').write_text(b.stdout+b.stderr);b.check_returncode()
with open('/tmp/hecbench-julia-study-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);q=subprocess.run([str(P/'full_dataset_probe'),str(P.parent/'original_assets/Dataset_1M.txt'),str(P)],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6'),timeout=180)
(P/'full_probe_run.log').write_text(q.stdout+q.stderr);q.check_returncode();files={n:{'bytes':(P/n).stat().st_size,'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest()} for n in ['actual_input_f32.bin','actual_output_f32.bin']};(P/'full_probe_receipt.json').write_text(json.dumps({'source_GPU_body_unchanged':True,'source_fast_math_flags':True,'n':1048576,'files':files,'scope':'DiagnosticactualfullGPUstate, independentfullfactorizationauditpending; not five-trialperformanceacceptance.'},indent=2)+'\n');print(q.stdout)
