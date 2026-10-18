"""Prepare a fresh four-model replication config, never launch measurements."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent;R=P.parents[1];x=json.loads((P/'study_config.json').read_text());old='/home/imo/HeCBench_julia'
def relocate(a):
 if isinstance(a,str):return str(R)+a[len(old):]if a.startswith(old)else a
 if isinstance(a,list):return [relocate(v)for v in a]
 if isinstance(a,dict):return {k:relocate(v)for k,v in a.items()}
 return a
x=relocate(x)
for key in list(x):
 if key.startswith('reuse_'):del x[key]
x['configuration_id']+='_cold_reproduction';dest=P/'study_config_cold.json';dest.write_text(json.dumps(x,indent=2)+'\n');print(dest)
