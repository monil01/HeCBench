#!/usr/bin/env python3
"""Relocate repository configs; preserve pinned external API/library prerequisites."""
from pathlib import Path
import argparse,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--native-deps',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cudacore-root',type=Path);p.add_argument('--jacc-root',type=Path);a=p.parse_args();R=a.source_root.resolve();P=R/'src/si-julia';a.output.mkdir(parents=True,exist_ok=True)
# Fixed vendor path is part of the measured source, not made silently relocatable.
lib=Path('/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224');assert lib.is_file()and hashlib.sha256(lib.read_bytes()).hexdigest()=='6571bf8344cf13f3b116e6d613dd234cefca081c2303c3667e71b25d0eba5ec2','Pinnedvendorlibrary/pathrequired; nofresh-hostexecutionclaim'
config=json.loads((P/'study_config_jacc_original_uniform1000_v2.json').read_text());oldR=str(Path(config['models']['cuda']['cwd']).parents[1]);olddeps='/tmp/si_native_readiness/build/_deps';oldcore='/home/imo/.julia/packages/CUDACore/NlVPI';oldjacc='/home/imo/.julia/packages/JACC/NLI8A'
# API SHA expectations come from saved binding receipts, so the old cache need not exist.
bound=json.loads((P/'final_source_binding_audit.json').read_text())['bindings']
def resolve_package(given,name,old,files):
 choices=[given]if given else list((Path.home()/'.julia/packages'/name).glob('*'))
 for root in choices:
  if all((root/f).is_file()and hashlib.sha256((root/f).read_bytes()).hexdigest()==bound[old+'/'+f]for f in files):return root.resolve()
 raise RuntimeError('No matching '+name+' API source; instantiatepinnedprojectandprovideexplicitroot')
core=resolve_package(a.cudacore_root,'CUDACore',oldcore,['src/array.jl','src/memory.jl','src/CUDAKernels.jl']);jacc=resolve_package(a.jacc_root,'JACC',oldjacc,['src/array.jl','ext/CUDAExt/CUDAExt.jl'])
def replace(x):
 if isinstance(x,str):return x.replace(oldR,str(R)).replace(olddeps,str(a.native_deps.resolve())).replace(oldcore,str(core)).replace(oldjacc,str(jacc))
 if isinstance(x,list):return[replace(v)for v in x]
 if isinstance(x,dict):return{k:replace(v)for k,v in x.items()}
 return x
for name in ['original_uniform1000','heterogeneous_full24','heterogeneous_partial19']:
 c=json.loads((P/f'study_config_jacc_{name}_v2.json').read_text());c.pop('reuse_models',None);c.pop('reuse_result',None);c=replace(c);f=a.output/f'si_cold_{name}_v2.json';f.write_text(json.dumps(c,indent=2)+'\n');print(f)
