#!/usr/bin/env python3
"""Prepare a relocatable fresh4model configuration after regenerating assets."""
from pathlib import Path
import argparse,json,subprocess,sys
p=Path(__file__).resolve().parent;r=p.parents[1];a=argparse.ArgumentParser();a.add_argument('--output',required=True,type=Path);args=a.parse_args()
subprocess.run([sys.executable,str(p/'prepare_assets.py')],check=True)
d=json.loads((p/'study_config.json').read_text());old='/home/imo/HeCBench_julia'
def relocate(x):
 if isinstance(x,str):return x.replace(old,str(r))
 if isinstance(x,list):return [relocate(v) for v in x]
 if isinstance(x,dict):return {k:relocate(v) for k,v in x.items() if not k.startswith('reuse_')}
 return x
d=relocate(d);args.output.write_text(json.dumps(d,indent=2)+'\n');print(args.output)
