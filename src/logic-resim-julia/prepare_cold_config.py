#!/usr/bin/env python3
from pathlib import Path
import argparse,json
P=Path(__file__).resolve().parent;R=P.parent.parent;a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();c=json.load(open(P/'study_config.json'));old=str(Path(c['models']['cuda']['cwd']).parents[1])
def convert(x):
 if isinstance(x,str)and x.startswith(old+'/'):return str(R)+x[len(old):]
 if isinstance(x,list):return[convert(z)for z in x]
 if isinstance(x,dict):return{k:convert(v)for k,v in x.items()}
 return x
c=convert(c);c.pop('reuse_result',None);c.pop('reuse_models',None);args.output.write_text(json.dumps(c,indent=2)+'\n');print(args.output)
