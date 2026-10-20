#!/usr/bin/env python3
"""Relocate a frozen four-model config, resolve exact JACC source files, remove reuse."""
import argparse,json,hashlib,os,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--jacc-package-root',type=Path);a=p.parse_args();B=Path(__file__).resolve().parent;c=json.loads((B/'study_config.json').read_text());old=str(Path(c['models']['cuda']['cwd']).parents[1]);new=str(a.source_root.resolve())
def relocate(v):
 if isinstance(v,str):return v.replace(old,new)
 if isinstance(v,list):return [relocate(x)for x in v]
 if isinstance(v,dict):return {k:relocate(x)for k,x in v.items()}
 return v
c=relocate(c);c={k:v for k,v in c.items()if not k.startswith('reuse_')};review=json.loads((B/'JACC_API_review.json').read_text());expected={x['path']:x['sha256']for x in review['actual_source_files']};resolved=[]
for dep in c['models']['jacc']['source_dependencies']:
 if dep not in expected:resolved.append(dep);continue
 suffix=dep.split('/NLI8A/',1)[1];candidates=[]
 if a.jacc_package_root:candidates.append(a.jacc_package_root/suffix)
 candidates.append(Path(dep));depots=[Path.home()/'.julia']+[Path(x)for x in os.environ.get('JULIA_DEPOT_PATH','').split(':')if x]
 for depot in depots:candidates.extend((depot/'packages/JACC').glob('*/'+suffix))
 match=next((f.resolve()for f in candidates if f.is_file()and hashlib.sha256(f.read_bytes()).hexdigest()==expected[dep]),None)
 if match is None:raise SystemExit('Missing exactpinnedJACCsource: '+suffix+'; do notsilentlysubstituteanotherbackendversion')
 resolved.append(str(match))
c['models']['jacc']['source_dependencies']=resolved;c['configuration_id']+='__cold_'+uuid.uuid4().hex[:12];c['cold_reproduction_note']='Freshall4fiveseries, samepredeclaredstrictfieldcontract; runaudit_saved_fields.pyafterward beforeacceptance; recordedGPU6/globalinternallock';a.output.write_text(json.dumps(c,indent=2)+'\n');print(a.output)
