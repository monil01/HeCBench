from pathlib import Path
import json,re
p=Path('/home/imo/HeCBench_julia/src/testSNAP-julia');d=p/'testSNAP-julia-JACC';d.mkdir(exist_ok=True);m=json.loads((p/'source_translation_manifest.json').read_text());a=['using CUDA','import JACC, Atomix','JACC.@init_backend','const MODEL="jacc"','include(joinpath(@__DIR__,"..","study_device.jl"))','@inline function snap_atomic!(a,i,v); Atomix.@atomic a[i] += v; nothing; end']
for f in m['functions']:
 if f['kind']!='kernel_body':continue
 name=f['name'];args=f['parameters'][:-2];oned=name in ('reset_ulisttot','set_ulisttot','reset_ylist');inds=['i']if oned else ['i','j']
 a+=['function '+name+'_entry('+','.join(inds+args)+')','ix=Int32(i)-Int32(1);iy='+('Int32(0)'if oned else 'Int32(j)-Int32(1)'),name+'('+','.join(args+['ix','iy'])+')','return nothing','end']
cu=(p/'testSNAP-julia-cuda.jl/main.jl').read_text();body=cu[cu.index('function launch_pipeline!'):cu.index('include(joinpath(@__DIR__,"..","study_common.jl"))')]
for line in body.splitlines():
 if 'CUDA.@cuda' not in line:a.append(line);continue
 mat=re.search(r'(\w+_entry)\((.*)\)$',line);name,args=mat.groups()
 if name=='reset_ulisttot_entry':nd='na*Int32(s.nu)';wg='256';blocks='cld('+nd+',256)';sizes=blocks+'*256'
 elif name=='reset_ylist_entry':nd='na*nd';wg='256';blocks='cld('+nd+',256)';sizes=blocks+'*256'
 elif name=='set_ulisttot_entry':nd='na';wg='256';blocks='cld(na,256)';sizes=blocks+'*256'
 elif name=='compute_yi_entry':wg='(16,16)';blocks='(cld(na,16),cld(nz,16))';sizes='(cld(na,16)*16,cld(nz,16)*16)'
 else:wg='(16,16)';blocks='pairgrid';sizes='(pairgrid[1]*16,pairgrid[2]*16)'
 a+=['spec=JACC.launch_spec(threads='+wg+',blocks='+blocks+',shmem_size=0,sync=false)','JACC.parallel_for(spec,'+sizes+','+name+','+args+')']
a+=['include(joinpath(@__DIR__,"..","study_common.jl"))','study()'];(d/'main.jl').write_text('\n'.join(a)+'\n')
c=json.loads((p/'study_config_ka.json').read_text());c['reuse_result']=str(p.parent.parent/'.porting-state/portable_julia/runs/20261020T045408953468Z_testSNAP/result.json');c['reuse_models']=['cuda','cuda_jl','ka'];spec=json.loads(json.dumps(c['models']['cuda_jl']));spec['command'][-1]=str(d/'main.jl');spec['command'][1]='--project=/home/imo/HeCBench_julia/src/_julia_jacc_env';c['models']['jacc']=spec;(p/'study_config_jacc.json').write_text(json.dumps(c,indent=2)+'\n');print('JACC seven ownentries withfreshshape-specific publiclaunch_spec/dynamicshared0')
