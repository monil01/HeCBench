#!/usr/bin/env python3
"""Create relocated fresh-run config; never alter archived configs or evidence."""
import argparse,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
d=json.loads((P/'study_config.json').read_text());oldroot=str(Path(d['models']['cuda']['cwd']).parent.parent)
def relocate(x):
 if isinstance(x,str):return str(ROOT)+x[len(oldroot):] if x.startswith(oldroot+'/') else x
 if isinstance(x,list):return [relocate(v) for v in x]
 if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
 return x
d=relocate(d)
for k in list(d):
 if k.startswith('reuse_'):del d[k]
d['configuration_id']+='__fresh_cold_reproduction'
a.output.write_text(json.dumps(d,indent=2)+'\n');print(a.output)
