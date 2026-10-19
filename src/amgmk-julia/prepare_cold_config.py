"""Relocate original AMGmk specs and remove saved-series reuse for a fresh measurement."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
c=json.loads((Path(__file__).parent/'study_config.json').read_text());old='/home/imo/HeCBench_julia';new=str(a.source_root.resolve())
def relocate(x):
 if isinstance(x,str):return x.replace(old,new)
 if isinstance(x,list):return [relocate(v) for v in x]
 if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
 return x
c=relocate(c)
for k in list(c):
 if k.startswith('reuse_'):del c[k]
a.output.write_text(json.dumps(c,indent=2)+'\n');print(a.output)
