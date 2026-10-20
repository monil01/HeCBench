from pathlib import Path
import ast,json
p=Path('src/lsqt-julia');tree=ast.parse((p/'generate_cuda_entries.py').read_text());entries=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='entries'for t in n.targets));d=p/'lsqt-julia-KA';d.mkdir(exist_ok=True)
s='''# GPUQT Julia translation, GPL-3.0-or-later; original notices in ../original_native/.
using CUDA, KernelAbstractions
const MODEL="ka"
const BACKEND=Val(:ka)
# Explicit CUDA-backend dependence in source-order shared/warp reductions.
@inline block_sync(::Val{:ka})=CUDA.sync_threads()
@inline warp_sync(::Val{:ka})=CUDA.sync_warp(UInt32(0xffffffff))
include("../study_device.jl")
'''
for name,args in entries:
 s+=f'@kernel function kernel_{name}!({args})\n'
 if name in ['ip1','ip2']:
  s+=' tid=Int32(@index(Local,Linear))-Int32(1);block=Int32(@index(Group,Linear))-Int32(1)\n sr=@localmem Float64 (256,)\n si=@localmem Float64 (256,)\n'
  s+=f' body_{name}!(tid,block,{args},sr,si,BACKEND)\n'
 else:s+=f' i=Int32(@index(Global,Linear))-Int32(1)\n body_{name}!(i,{args})\n'
 s+=' nothing\nend\n'
for code,(name,args)in enumerate(entries):
 s+=f'function dispatch!(::Val{{:{name}}},s,blocks,{args})\n record_launch!(s,{code})\n'
 if name=='cheb01':s+=' record_scalar!(s,9,(b0,b1,Float64(direction)))\n'
 if name=='cheb2':s+=' record_scalar!(s,10,(b,Float64(label)))\n'
 if name=='cheb1x':s+=' record_scalar!(s,11,(b,))\n'
 if name=='cheb2x':s+=' record_scalar!(s,12,(b,Float64(label)))\n'
 s+=f' kernel_{name}!(CUDABackend(),256)({args};ndrange=blocks*256)\n nothing\nend\n'
s+='include("../study_common.jl")\nmain()\n';(d/'main.jl').write_text(s);(p/'generate_ka_entries.py').write_text(Path(__file__).read_text())
c=json.loads((p/'study_config_cuda_jl.json').read_text());c['reuse_result']=str(Path('.porting-state/portable_julia/runs/20261020T000258223416Z_lsqt/result.json').resolve());c['reuse_models']=['cuda','cuda_jl'];spec=dict(c['models']['cuda_jl']);spec['command']=list(spec['command']);spec['command'][3]=str((d/'main.jl').resolve());c['models']['ka']=spec;(p/'study_config_ka.json').write_text(json.dumps(c,indent=2)+'\n')
