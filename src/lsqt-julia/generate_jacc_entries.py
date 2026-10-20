from pathlib import Path
import ast,json
p=Path('src/lsqt-julia');tree=ast.parse((p/'generate_cuda_entries.py').read_text());entries=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='entries'for t in n.targets));d=p/'lsqt-julia-JACC';d.mkdir(exist_ok=True)
s='''# GPUQT Julia translation, GPL-3.0-or-later; original notices in ../original_native/.
using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
const BACKEND=Val(:jacc)
@inline block_sync(::Val{:jacc})=JACC.sync_workgroup()
# Explicit CUDA-backend dependency for legacy source warp-order reductions.
@inline warp_sync(::Val{:jacc})=CUDA.sync_warp(UInt32(0xffffffff))
include("../study_device.jl")
'''
for name,args in entries:
 s+=f'function kernel_{name}!(index,{args})\n'
 if name in ['ip1','ip2']:
  s+=' i=Int32(index)-Int32(1);tid=rem(i,Int32(256));block=div(i,Int32(256))\n sr=CUDA.CuStaticSharedArray(Float64,256);si=CUDA.CuStaticSharedArray(Float64,256)\n'
  s+=f' body_{name}!(tid,block,{args},sr,si,BACKEND)\n'
 else:s+=f' i=Int32(index)-Int32(1)\n body_{name}!(i,{args})\n'
 s+=' nothing\nend\n'
for code,(name,args)in enumerate(entries):
 s+=f'function dispatch!(::Val{{:{name}}},s,blocks,{args})\n record_launch!(s,{code})\n'
 if name=='cheb01':s+=' record_scalar!(s,9,(b0,b1,Float64(direction)))\n'
 if name=='cheb2':s+=' record_scalar!(s,10,(b,Float64(label)))\n'
 if name=='cheb1x':s+=' record_scalar!(s,11,(b,))\n'
 if name=='cheb2x':s+=' record_scalar!(s,12,(b,Float64(label)))\n'
 s+=f' spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)\n JACC.parallel_for(spec,blocks*256,kernel_{name}!,{args})\n nothing\nend\n'
s+='include("../study_common.jl")\nmain()\n';(d/'main.jl').write_text(s);(p/'generate_jacc_entries.py').write_text(Path(__file__).read_text())
c=json.loads((p/'study_config_ka.json').read_text());c['reuse_result']=str(Path('.porting-state/portable_julia/runs/20261020T000614307167Z_lsqt/result.json').resolve());c['reuse_models']=['cuda','cuda_jl','ka'];spec=dict(c['models']['cuda_jl']);spec['command']=list(spec['command']);spec['command'][2]='--project='+str(Path('src/_julia_jacc_env').resolve());spec['command'][3]=str((d/'main.jl').resolve());c['models']['jacc']=spec;(p/'study_config_jacc.json').write_text(json.dumps(c,indent=2)+'\n')
