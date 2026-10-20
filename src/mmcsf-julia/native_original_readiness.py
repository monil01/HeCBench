#!/usr/bin/env python3
"""Original native readiness; caller owns GPU lock. No system changes."""
import pathlib,subprocess,json,hashlib,time
B=pathlib.Path(__file__).resolve().parent;R=B.parent.parent;H=B/'reference_history';L=R/'.porting-state/porting_logs/mmcsf-julia-shared';L.mkdir(parents=True,exist_ok=True);events=[]
commands=[('original_build',['make','-j4','ARCH=sm_70','BOOST=/usr/include']),('original_toy',['./main','-i','toy.tns','-m','0','-R','32','-f','128','-w','1']),('original_toy_initcheck',['/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/compute-sanitizer/compute-sanitizer','--tool','initcheck','--error-exitcode','17','./main','-i','toy.tns','-m','0','-R','32','-f','128','-w','1'])]
for label,cmd in commands:
 t=time.monotonic()
 with (L/(label+'.log')).open('w')as f:x=subprocess.run(cmd,cwd=H,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 events.append({'label':label,'command':cmd,'returncode':x.returncode,'elapsed_seconds':time.monotonic()-t,'log':str((L/(label+'.log')).relative_to(R))})
 if label=='original_build'and x.returncode:break
p=B/'native_original_readiness.json';q=p.with_suffix('.tmp');q.write_text(json.dumps({'benchmark':'mmcsf','events':events,'scope':'Readiness diagnostics, not five-trial certification. Originalsource unchanged; existing systemBoostheaders used, no installation.'},indent=2)+'\n');q.replace(p);print(json.dumps(events))
