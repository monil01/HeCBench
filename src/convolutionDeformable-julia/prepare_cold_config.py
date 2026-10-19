#!/usr/bin/env python3
"""Relocate a fresh four-model kernel-extension contract without reusing saved timings."""
from pathlib import Path
import json,sys,hashlib
P=Path(__file__).resolve().parent;R=P.parent.parent;c=json.loads((P/'study_config.json').read_text());old=c['models']['cuda']['cwd'].removesuffix('/src/convolutionDeformable-julia')
def relocate(x):
 if isinstance(x,str):return x.replace(old,str(R))
 if isinstance(x,list):return [relocate(v)for v in x]
 if isinstance(x,dict):return {k:relocate(v)for k,v in x.items()}
 return x
c=relocate(c);c.pop('reuse_result',None);c.pop('reuse_models',None)
# Package dependencies are exact source bindings, not paths assumed to share this user's home.
audit=json.loads((P/'saved_output_audit.json').read_text())
for spec in c['models'].values():
 deps=spec['source_dependencies']
 for k,name in enumerate(deps):
  if '/.julia/packages/'not in name or Path(name).exists():continue
  package,slug,*relative=name.split('/.julia/packages/',1)[1].split('/');expected=audit['source_bindings'][name];matches=[q for q in (Path.home()/'.julia/packages'/package).glob('*/'+'/'.join(relative))if hashlib.sha256(q.read_bytes()).hexdigest()==expected]
  if len(matches)!=1:raise RuntimeError('Instantiate pinned environment; expected exact source package dependency '+name)
  deps[k]=str(matches[0])
out=Path(sys.argv[1])if len(sys.argv)>1 else P/'study_config_cold.local.json';out.write_text(json.dumps(c,indent=2)+'\n');print(out)
