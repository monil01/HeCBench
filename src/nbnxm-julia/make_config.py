#!/usr/bin/env python3
"""Generate relocated full-pipeline configurations; no CUDA work here."""
import argparse,pathlib,json
p=argparse.ArgumentParser();p.add_argument('--models',choices=['native','cuda_jl','ka','jacc','all'],default='all');p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--canonical-dir',type=pathlib.Path);p.add_argument('--aux-dir',type=pathlib.Path);a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;R=B.parent.parent
text=(B/'configuration_template.json').read_text().replace('{repo_root}',str(R));c=json.loads(text);keep={'native':['cuda'],'cuda_jl':['cuda','cuda_jl'],'ka':['cuda','cuda_jl','ka'],'jacc':['cuda','cuda_jl','jacc'],'all':['cuda','cuda_jl','ka','jacc']}[a.models];c['models']={k:v for k,v in c['models'].items()if k in keep}
old=R/'.porting-state/porting_logs/nbnxm-julia-shared/canonical_inputs_v1'
if a.canonical_dir:
 text=json.dumps(c).replace(str(old),str(a.canonical_dir.resolve()));c=json.loads(text)
 for asset in c['input_contract']['actual_input_readback_assets']:asset['path']=str(a.canonical_dir.resolve()/pathlib.Path(asset['path']).name)
if a.aux_dir:
 for k,m in c['models'].items():m['command'][-1]=str(a.aux_dir.resolve()/k)
for m in c['models'].values():
 for f in m['source_dependencies']:
  if not pathlib.Path(f).is_file():raise FileNotFoundError(f)
a.output.parent.mkdir(parents=True,exist_ok=True);q=a.output.with_suffix('.tmp');q.write_text(json.dumps(c,indent=2)+'\n');q.replace(a.output);print(a.output)
