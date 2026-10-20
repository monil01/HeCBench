#!/usr/bin/env python3
"""Relocate frozen measurement specification, run all4 afresh without archived reuse."""
from pathlib import Path
import argparse,json
P=Path(__file__).resolve().parent;R=P.parent.parent;a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();c=json.load(open(P/'study_config.json'));old=Path(c['models']['cuda']['cwd']).parents[1]
def relocate(x):
 if isinstance(x,str)and x.startswith(str(old)+'/'):return str(R)+x[len(str(old)):]
 if isinstance(x,list):return[relocate(v)for v in x]
 if isinstance(x,dict):return{k:relocate(v)for k,v in x.items()}
 return x
c=relocate(c);c.pop('reuse_result',None);c.pop('reuse_models',None);args.output.write_text(json.dumps(c,indent=2)+'\n');print(args.output)
