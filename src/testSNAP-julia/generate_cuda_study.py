from pathlib import Path
import json
p=Path('/home/imo/HeCBench_julia/src/testSNAP-julia');m=json.loads((p/'source_translation_manifest.json').read_text());d=p/'testSNAP-julia-cuda.jl';d.mkdir(exist_ok=True)
a=['using CUDA','const MODEL="cuda_jl"','include(joinpath(@__DIR__,"..","study_device.jl"))','@inline function snap_atomic!(a,i,v); CUDA.@atomic a[i] += v; nothing; end']
for f in m['functions']:
 if f['kind']!='kernel_body':continue
 name=f['name'];args=f['parameters'][:-2];a+=['function '+name+'_entry('+','.join(args)+')','ix=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)','iy=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)',name+'('+','.join(args+['ix','iy'])+')','return nothing','end']
a+=['function launch_pipeline!(c)','s=c.s;na=Int32(s.na);nb=Int32(s.nb);J=Int32(s.J);nd=Int32(s.nd);pdim=Int32(s.pdim);nz=Int32(s.nz);jdim=Int32(s.jdim);pairgrid=(cld(na,16),cld(nb,16))',
'CUDA.@cuda threads=256 blocks=cld(na*Int32(s.nu),256) reset_ulisttot_entry(c.ut.d,na*Int32(s.nu))',
'CUDA.@cuda threads=256 blocks=cld(na,256) set_ulisttot_entry(c.ut.d,c.iu.d,na,J,s.s[2])',
'CUDA.@cuda threads=(16,16) blocks=pairgrid update_ulisttot_entry(c.r.d,c.rc.d,c.w.d,c.parity.d,c.iu.d,c.root.d,c.u.d,c.ut.d,na,nb,Int32(1),J,pdim)','CUDA.device_synchronize()',
'CUDA.@cuda threads=256 blocks=cld(na*nd,256) reset_ylist_entry(c.y.d,na*nd)',
'CUDA.@cuda threads=(16,16) blocks=(cld(na,16),cld(nz,16)) compute_yi_entry(c.iz.d,c.zb.d,c.cg.d,c.icg.d,c.iu.d,c.id.d,c.ut.d,c.y.d,na,nz,jdim)','CUDA.device_synchronize()',
'CUDA.@cuda threads=(16,16) blocks=pairgrid compute_duidrj_entry(c.w.d,c.r.d,c.rc.d,c.root.d,c.u.d,c.du.d,na,nb,J,nd,pdim,Int32(1))','CUDA.device_synchronize()',
'CUDA.@cuda threads=(16,16) blocks=pairgrid compute_deidrj_entry(c.id.d,c.du.d,c.y.d,c.d.d,na,nb,J,nd)','CUDA.device_synchronize()','nothing','end','include(joinpath(@__DIR__,"..","study_common.jl"))','study()']
(d/'main.jl').write_text('\n'.join(a)+'\n')
c=json.loads((p/'study_config_native.json').read_text());c['reuse_result']=str(p.parent.parent/'.porting-state/portable_julia/runs/20261020T043914250988Z_testSNAP/result.json');c['reuse_models']=['cuda'];base=c['models']['cuda'];deps=base['source_dependencies']+[str(p/x) for x in ['study_device.jl','study_common.jl','translate_source.py','source_translation_manifest.json']];c['models']['cuda_jl']={'structured':True,'cwd':str(p),'bind_all_declared_dependencies':True,'runtime_assets':base['runtime_assets'],'source_dependencies':deps,'build':['g++','-O3','-std=c++17','-fPIC','-shared',str(p/'study_observer.cpp'),'-lcrypto','-lz','-o','study_observer.so'],'command':['/home/imo/.juliaup/bin/julia','--project=/home/imo/HeCBench_julia/src/_julia_ka_env','--startup-file=no',str(d/'main.jl')],'verification_method':base['verification_method']};(p/'study_config_cuda_jl.json').write_text(json.dumps(c,indent=2)+'\n');print('cuda authored',len(a),'lines')
