#!/usr/bin/env python3
"""Relocate a fresh named run; never mutate frozen archived configs/results."""
from pathlib import Path
import argparse,json,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--configuration',choices=['primary','partial_corrected'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads((P/f'study_config_{a.configuration}.json').read_text());oldroot=str(Path(d['models']['cuda']['cwd']).parent.parent)
def relocate(x):
 if isinstance(x,str):return str(ROOT)+x[len(oldroot):] if x.startswith(oldroot+'/') else x
 if isinstance(x,list):return [relocate(v) for v in x]
 if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
 return x
d=relocate(d)
for k in list(d):
 if k.startswith('reuse_'):del d[k]
# The explicit private JACC wrapper is version-bound. Resolve its actual local
# package path by exact file hash; do not accept a different implementation.
expected=json.loads((P/'final_source_binding_audit.json').read_text())['bindings']
for spec in d['models'].values():
 deps=[]
 for name in spec.get('source_dependencies',[]):
  if '/.julia/packages/JACC/' in name:
   oldkey=next(k for k in expected if '/.julia/packages/JACC/' in k and k.endswith('/ext/CUDAExt/CUDAExt.jl'))
   want=expected[oldkey];candidates=[Path(name),*sorted((Path.home()/'.julia/packages/JACC').glob('*/ext/CUDAExt/CUDAExt.jl'))]
   match=next((q for q in candidates if q.is_file() and hashlib.sha256(q.read_bytes()).hexdigest()==want),None)
   if match is None:raise RuntimeError('Instantiate pinned JACC environment and retain exact CUDAExt source; no matching dependency found')
   name=str(match.resolve())
  deps.append(name)
 spec['source_dependencies']=deps
d['configuration_id']+='__fresh_cold_reproduction';a.output.write_text(json.dumps(d,indent=2)+'\n');print(a.output)
