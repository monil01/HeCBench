#!/usr/bin/env python3
"""New all-four reproduction config; drops historical capture reuse explicitly."""
import argparse,json,pathlib,subprocess,shutil
p=argparse.ArgumentParser();p.add_argument('--source-root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[2]);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--julia',default=shutil.which('julia') or 'julia');p.add_argument('--openblas-library',type=pathlib.Path);a=p.parse_args();root=a.source_root.resolve();parent=root/'src/hpl-julia'
x=json.loads((pathlib.Path(__file__).parent/'study_config_recorded.json').read_text());old='/home/imo/HeCBench_julia'
def remap(o):
 if isinstance(o,str):return o.replace(old,str(root))
 if isinstance(o,list):return [remap(v) for v in o]
 if isinstance(o,dict):return {k:remap(v) for k,v in o.items()}
 return o
x=remap(x)
for k in list(x):
 if k.startswith('reuse_'):del x[k]
lib=a.openblas_library or pathlib.Path(subprocess.check_output([a.julia,'--startup-file=no','-e','print(realpath(joinpath(Sys.BINDIR,"..","lib","julia","libopenblas64_.so")))'],text=True));assert lib.is_file()
for model,s in x['models'].items():
 script=s['build'][1];output=s['build'][2];s['build']=['python3',str(parent/'relocate_build.py'),'--build-script',script,'--openblas-library',str(lib.resolve()),'--output',output]
 s['source_dependencies'] += [str(parent/'relocate_build.py'),str(lib.resolve())]
 if model!='cuda':s['command'][2]=a.julia
x['environment']['reproduction_relocation']={'openblas_library':str(lib.resolve()),'julia':a.julia,'qualification':'Newsourceboundall4measurements; explicitloaderpathrelocationonly, doesnotretroactivelyalteroriginalsourcehashes/trials.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(x,indent=2)+'\n');print(a.output)
