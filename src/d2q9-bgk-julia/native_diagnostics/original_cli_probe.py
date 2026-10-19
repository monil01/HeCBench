#!/usr/bin/env python3
from pathlib import Path
import subprocess,os,json,hashlib,fcntl,time
P=Path(__file__).resolve().parent;R=P.parents[2];source=R/'d2q9-bgk-cuda/main.cu'
# Resolve repository source independently of the shell's working directory.
source=P.parent.parent/'d2q9-bgk-cuda/main.cu'
build=subprocess.run(['nvcc','-O3','-std=c++17','-arch=sm_70',str(source),'-o',str(P/'original_cli')],capture_output=True,text=True);(P/'build.log').write_text(build.stdout+build.stderr);build.check_returncode()
out=P/'fresh_cli_outputs';out.mkdir(exist_ok=True)
for name in ['input.params','obstacles.dat']:(out/name).write_bytes((P/name).read_bytes())
with open('/tmp/hecbench-julia-study-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='6');run=subprocess.run([str(P/'original_cli'),'input.params','obstacles.dat'],cwd=out,env=env,capture_output=True,text=True,timeout=120);(P/'run.log').write_text(run.stdout+run.stderr);run.check_returncode()
 assert '==done==' in run.stdout
 outputs={n:{'bytes':(out/n).stat().st_size,'sha256':hashlib.sha256((out/n).read_bytes()).hexdigest()} for n in ['final_state.dat','av_vels.dat']}
 (P/'original_cli_receipt.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'exit_code':run.returncode,'actual_named_inputs':['input.params','obstacles.dat'],'outputs':outputs,'scope':'Untouched original CLI readiness, named synthetic1100step input; output numerical validation separate. Original DVC fixtures unavailable. Timer actually101launches dividedby100; preserve source output label and do not use it as corrected mean.'},indent=2)+'\n')
print(json.dumps(outputs))
