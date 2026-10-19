"""Generate a relocated fresh-study config; historical measurements are not changed."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();B=Path(__file__).resolve().parent;R=B.parent.parent;d=json.loads((B/'study_configuration.json').read_text());old='/home/imo/HeCBench_julia';dataold=old+'/.porting-state/portable_julia/external_assets/egs/data'
def relocate(x):
 if isinstance(x,dict):return {k:relocate(v)for k,v in x.items()}
 if isinstance(x,list):return [relocate(v)for v in x]
 if isinstance(x,str):return x.replace(dataold,str(a.data_root.resolve())).replace(old,str(R))
 return x
d=relocate(d);d.pop('reuse_result',None);d.pop('reuse_models',None)
for spec in d['models'].values():spec['source_dependencies']=[x for x in spec['source_dependencies']if Path(x).name not in ['CUDA_EGS_phantom.pegs4dat','CUDA_EGS_phantom_32.egsphant']]
a.output.write_text(json.dumps(d,indent=2)+'\n');print(a.output)
