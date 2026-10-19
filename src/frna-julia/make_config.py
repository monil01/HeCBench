"""Relocate frozen model specs and request a fresh four-model series."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();B=Path(__file__).resolve().parent;R=B.parents[1];x=json.loads((B/'study_config_recorded.json').read_text());old='/home/imo/HeCBench_julia'
def walk(v):
 if isinstance(v,str):return v.replace(old,str(R))
 if isinstance(v,list):return [walk(z)for z in v]
 if isinstance(v,dict):return {k:walk(z)for k,z in v.items()}
 return v
x=walk(x);x.pop('reuse_result',None);x.pop('reuse_models',None);a.output.write_text(json.dumps(x,indent=2)+'\n');print(a.output)
