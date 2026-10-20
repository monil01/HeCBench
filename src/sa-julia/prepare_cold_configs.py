#!/usr/bin/env python3
"""Relocate full four-model SA configs without reusing measured trial series."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();R=a.source_root.resolve();P=R/'src/sa-julia';a.output.mkdir(parents=True,exist_ok=True)
for mode in ['primary','controls']:
 c=json.loads((P/f'study_config_jacc_{mode}.json').read_text());old=str(Path(c['models']['cuda']['cwd']).parents[1]);c.pop('reuse_result',None);c.pop('reuse_models',None)
 def replace(x):
  if isinstance(x,str):return x.replace(old,str(R))
  if isinstance(x,list):return[replace(v)for v in x]
  if isinstance(x,dict):return{k:replace(v)for k,v in x.items()}
  return x
 c=replace(c);f=a.output/f'sa_cold_{mode}.json';f.write_text(json.dumps(c,indent=2)+'\n');print(f)
