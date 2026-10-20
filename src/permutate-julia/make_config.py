#!/usr/bin/env python3
"""Relocate unmeasured reproduction configuration without changing frozen runs."""
import argparse,pathlib,json
p=argparse.ArgumentParser();p.add_argument('--models',choices=['native','cuda_jl','ka','jacc','all'],default='all');p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--canonical-dir',type=pathlib.Path);p.add_argument('--asset',type=pathlib.Path);p.add_argument('--aux-dir',type=pathlib.Path);a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;R=B.parents[1];text=(B/'configuration_template.json').read_text().replace('{repo_root}',str(R))
if a.canonical_dir:text=text.replace(str(R/'.porting-state/porting_logs/permutate-julia-shared/canonical_inputs_v1'),str(a.canonical_dir.resolve()))
if a.asset:text=text.replace(str(R/'.porting-state/porting_logs/permutate-julia-shared/assets/truerand_1bit.bin'),str(a.asset.resolve()))
c=json.loads(text);keep={'native':['cuda'],'cuda_jl':['cuda','cuda_jl'],'ka':['cuda','cuda_jl','ka'],'jacc':['cuda','cuda_jl','jacc'],'all':['cuda','cuda_jl','ka','jacc']}[a.models];c['models']={k:v for k,v in c['models'].items()if k in keep}
if a.aux_dir:
 for name,m in c['models'].items():m['command'][-1]=str(a.aux_dir.resolve()/name)
for m in c['models'].values():
 for name in m['source_dependencies']:
  if not pathlib.Path(name).is_file():raise FileNotFoundError(name)
a.output.parent.mkdir(parents=True,exist_ok=True);q=a.output.with_suffix('.tmp');q.write_text(json.dumps(c,indent=2)+'\n');q.replace(a.output);print(a.output)
