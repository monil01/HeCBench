#!/usr/bin/env python3
from pathlib import Path
import json,sys
P=Path(__file__).resolve().parent;R=P.parent.parent;c=json.loads((P/'study_config.json').read_text());old=c['models']['cuda']['cwd'].removesuffix('/src/gibbs-julia')
def relocate(x):
 if isinstance(x,str):return x.replace(old,str(R))
 if isinstance(x,list):return [relocate(y)for y in x]
 if isinstance(x,dict):return {k:relocate(v)for k,v in x.items()}
 return x
c=relocate(c);c.pop('reuse_result',None);c.pop('reuse_models',None);out=Path(sys.argv[1])if len(sys.argv)>1 else P/'study_config_cold.local.json';out.write_text(json.dumps(c,indent=2)+'\n');print(out)
