#!/usr/bin/env python3
"""Create a fresh all4 config; historical capture reuse is deliberately removed."""
import pathlib,json,argparse,shutil
p=argparse.ArgumentParser();p.add_argument('--source-root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[2]);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--julia',default=shutil.which('julia') or 'julia');a=p.parse_args();r=a.source_root.resolve();x=json.loads((pathlib.Path(__file__).parent/'study_config_recorded.json').read_text())
def remap(o):
 if isinstance(o,str):return o.replace('/home/imo/HeCBench_julia',str(r))
 if isinstance(o,list):return [remap(v) for v in o]
 if isinstance(o,dict):return {k:remap(v) for k,v in o.items()}
 return o
x=remap(x)
for key in list(x):
 if key.startswith('reuse_'):del x[key]
for model,s in x['models'].items():
 if model!='cuda':s['command'][0]=a.julia
x['reproduction_scope']='Freshall4five-trialseries on explicitlocalenvironment; priorverifiedresults/sources notaltered.';a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(x,indent=2)+'\n');print(a.output)
