#!/usr/bin/env python3
"""Relocate frozen protocol, rebuild host runtime, remove historical GPU-series reuse."""
from pathlib import Path
import argparse,json,subprocess
P=Path(__file__).resolve().parent;R=P.parents[1];a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);x=a.parse_args();subprocess.run(['python3',str(P/'build_runtime.py')],check=True);c=json.load(open(P/'study_config.json'));old=Path(c['models']['cuda']['cwd']).parents[1]
def relocate(v):
 if isinstance(v,str)and v.startswith(str(old)+'/'):return str(R)+v[len(str(old)):]
 if isinstance(v,list):return[relocate(z)for z in v]
 if isinstance(v,dict):return{k:relocate(z)for k,z in v.items()}
 return v
c=relocate(c);c.pop('reuse_result',None);c.pop('reuse_models',None);x.output.write_text(json.dumps(c,indent=2)+'\n');print(x.output)
