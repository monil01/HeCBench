#!/usr/bin/env python3
from pathlib import Path
import json,subprocess,sys
p=Path(__file__).resolve().parent;r=p.parents[1];d=json.loads((p/'study_config.json').read_text());old=d['models']['cuda']['cwd'].split('/src/floydwarshall2-julia')[0]
def relocate(x):
 if isinstance(x,str):return x.replace(old,str(r))
 if isinstance(x,list):return [relocate(v) for v in x]
 if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
 return x
d=relocate(d)
for k in ('reuse_models','reuse_result'):d.pop(k,None)
subprocess.run([sys.executable,str(p/'generate_assets.py')],cwd=p,check=True)
out=p/'study_config_cold_generated.json';out.write_text(json.dumps(d,indent=2)+'\n');print(out)
