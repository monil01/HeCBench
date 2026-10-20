#!/usr/bin/env python3
"""Recreate the named full original MD study on a relocated checkout."""
import pathlib,argparse,json,copy
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--models',choices=['native','cuda_jl','ka','jacc','all'],default='all');a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;R=B.parent.parent
c=json.loads((B/'native_configuration_template.json').read_text().replace('{benchmark_parent}',str(B)))
for model,child,env in [('cuda_jl','mcmd-julia-cuda.jl','_julia_ka_env'),('ka','mcmd-julia-KA','_julia_ka_env'),('jacc','mcmd-julia-JACC','_julia_jacc_env')]:
 if model=='cuda_jl':
  if a.models=='native':continue
 elif a.models not in ('all',model):continue
 m=copy.deepcopy(c['models']['cuda']);m['cwd']=str(B/child)
 m['build']=['nvcc','-std=c++17','-O3','-arch={arch}','-D_MWAITXINTRIN_H_INCLUDED','-D_FORCE_INLINES','-D__STRICT_ANSI__','-DGPU','-DMCMD_SHARED','-shared','-Xcompiler','-fPIC',str(B/'full_study.cpp'),str(B/'force_dispatch.cu'),'-o','study_gpu.so']
 m['command']=['julia','--project='+str(R/'src'/env),str(B/child/'main.jl'),'study_gpu.so',str(B)]
 m['source_dependencies'] += [str(B/n)for n in ['mcmd_device.jl','study_common.jl']]+[str(B/child/'main.jl')]+[str(R/'src'/env/n)for n in ['Project.toml','Manifest.toml','LocalPreferences.toml']]
 m['native_check_description']+=' AuthoredselectedJuliaforceworkitemwithinoriginalCPUfullMDcontrollerCABI; CUDAdeviceatomic/libdeviceextensions, no nativekernel fallback.'
 c['models'][model]=m
for model,m in c['models'].items():
 for d in m['source_dependencies']:
  if not pathlib.Path(d).is_file():raise FileNotFoundError((model,d))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(c,indent=2)+'\n');print(a.output)
