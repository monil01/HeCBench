#!/usr/bin/env python3
"""Rebuild host-only oracle and prepare a fresh four-model verification config."""
from pathlib import Path
import json,subprocess
p=Path(__file__).resolve().parent;r=p.parents[1]
c=json.loads((p/'study_config.json').read_text());old=c['models']['cuda']['cwd'].split('/src/perlin-julia')[0]
def relocate(x):
    if isinstance(x,str):return x.replace(old,str(r))
    if isinstance(x,list):return [relocate(v) for v in x]
    if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
    return x
c=relocate(c)
for k in ('reuse_models','reuse_result'):c.pop(k,None)
subprocess.run(json.loads((p/'checker_build_recipe.json').read_text()),cwd=p,check=True)
out=p/'study_config_cold_generated.json';out.write_text(json.dumps(c,indent=2)+'\n')
print(out)
