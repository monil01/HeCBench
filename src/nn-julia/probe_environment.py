"""Postmeasurement environment evidence only; serialize outside runner understudyGPUlock."""
from pathlib import Path
import subprocess,json,hashlib,os
P=Path(__file__).resolve().parent;R=P.parent.parent;D=R/'.porting-state/portable_julia/diagnostics/nn';out={}
code='using CUDA,Libdl;println("JULIA_VERSION=",VERSION);println("CUDA_RUNTIME=",CUDA.runtime_version());println("CUDA_DRIVER=",CUDA.driver_version());println("DEVICE=",CUDA.device());CUDA.versioninfo();for p in Libdl.dllist();occursin(r"libcuda|libnvrtc|libLLVM",p)&&println("LOADED_LIBRARY=",p);end'
for env in ['_julia_ka_env','_julia_jacc_env']:
 project=R/'src'/env;cmd=['julia','--project='+str(project),'-e',code];run=subprocess.run(cmd,capture_output=True,text=True);log=D/(env+'_probe.log');log.write_text(run.stdout+run.stderr)
 assert run.returncode==0
 fields=dict(line.split('=',1)for line in run.stdout.splitlines()if any(line.startswith(k+'=')for k in ['JULIA_VERSION','CUDA_RUNTIME','CUDA_DRIVER','DEVICE']))
 out[env]={'command':cmd,'exit_status':run.returncode,'fields':fields,'log':{'path':str(log.relative_to(R)),'sha256':hashlib.sha256(log.read_bytes()).hexdigest()},'project_bindings':{f:hashlib.sha256((project/f).read_bytes()).hexdigest()for f in ['Project.toml','Manifest.toml','LocalPreferences.toml']if(project/f).exists()},'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'scope':'Actualpostmeasurementprobeofsameprojectenvironments; noauthoredbenchmarkkernel/ROIexecution. Pinnedmanifest/prefs snapshot, notclaiminprocessenvprintinearlierrows.'}
(P/'environment_snapshot.json').write_text(json.dumps(out,indent=2)+'\n');print('NN_ENVIRONMENT_PROBE_PASS',json.dumps({k:v['fields']for k,v in out.items()}))
