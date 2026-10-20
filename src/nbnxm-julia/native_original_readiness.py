#!/usr/bin/env python3
"""Original source readiness only; caller owns GPU6 lock."""
import pathlib,subprocess,json,time
B=pathlib.Path(__file__).resolve().parent;R=B.parent.parent;H=B/'reference_history';L=R/'.porting-state/porting_logs/nbnxm-julia-shared';events=[]
for label,cmd in [('original_build',['make','-j2','ARCH=sm_70']),('original_full10000',['./main','10000'])]:
 t=time.monotonic()
 with (L/(label+'.log')).open('w')as f:p=subprocess.run(cmd,cwd=H,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 events.append({'label':label,'command':cmd,'returncode':p.returncode,'elapsed_seconds':time.monotonic()-t,'log':str((L/(label+'.log')).relative_to(R))})
 if p.returncode:break
q=B/'native_original_readiness.tmp';q.write_text(json.dumps({'benchmark':'nbnxm','events':events,'scope':'Originalbuild/full10000repeatbothphasesreadiness, not5checkedtrials. Sourceimaskallzero disablespairforcearithmetic; no scientificforceaccuracyclaim.'},indent=2)+'\n');q.replace(B/'native_original_readiness.json');print(json.dumps(events))
