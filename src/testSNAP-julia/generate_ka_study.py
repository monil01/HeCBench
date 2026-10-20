from pathlib import Path
import json,re
p=Path('/home/imo/HeCBench_julia/src/testSNAP-julia');d=p/'testSNAP-julia-KA';d.mkdir(exist_ok=True);m=json.loads((p/'source_translation_manifest.json').read_text());a=['using CUDA, KernelAbstractions','import Atomix','const MODEL="ka"','const BACKEND=CUDA.CUDABackend()','include(joinpath(@__DIR__,"..","study_device.jl"))','@inline function snap_atomic!(a,i,v); Atomix.@atomic a[i] += v; nothing; end']
for f in m['functions']:
 if f['kind']!='kernel_body':continue
 name=f['name'];args=f['parameters'][:-2];oned=name in ('reset_ulisttot','set_ulisttot','reset_ylist')
 a+=['@kernel function '+name+'_entry('+','.join(args)+')']
 if oned:a+=['ix=Int32(@index(Global,Linear))-Int32(1);iy=Int32(0)']
 else:a+=['index=@index(Global,NTuple);ix=Int32(index[1])-Int32(1);iy=Int32(index[2])-Int32(1)']
 a+=[name+'('+','.join(args+['ix','iy'])+')','end']
cu=(p/'testSNAP-julia-cuda.jl/main.jl').read_text();body=cu[cu.index('function launch_pipeline!'):cu.index('include(joinpath(@__DIR__,"..","study_common.jl"))')]
lines=[]
for line in body.splitlines():
 if 'CUDA.@cuda' not in line:lines.append(line);continue
 mat=re.search(r'(\w+_entry)\((.*)\)$',line);name,args=mat.groups()
 if name=='reset_ulisttot_entry':nd='na*Int32(s.nu)';wg='256'
 elif name=='reset_ylist_entry':nd='na*nd';wg='256'
 elif name=='set_ulisttot_entry':nd='na';wg='256'
 elif name=='compute_yi_entry':nd='(na,nz)';wg='(16,16)'
 else:nd='(na,nb)';wg='(16,16)'
 lines.append(name+'(BACKEND,'+wg+')('+args+';ndrange='+nd+')')
a+=lines+['include(joinpath(@__DIR__,"..","study_common.jl"))','study()'];(d/'main.jl').write_text('\n'.join(a)+'\n')
c=json.loads((p/'study_config_cuda_jl.json').read_text());c['reuse_result']=str(p.parent.parent/'.porting-state/portable_julia/runs/20261020T044806538805Z_testSNAP/result.json');c['reuse_models']=['cuda','cuda_jl'];spec=json.loads(json.dumps(c['models']['cuda_jl']));spec['command'][-1]=str(d/'main.jl');c['models']['ka']=spec;(p/'study_config_ka.json').write_text(json.dumps(c,indent=2)+'\n');print('KAownentries',sum(f['kind']=='kernel_body'for f in m['functions']))
