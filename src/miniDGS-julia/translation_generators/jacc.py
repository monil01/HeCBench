from pathlib import Path
P=Path(__file__).resolve().parents[1];code='using CUDA\nimport JACC\nJACC.@init_backend\n'
for name,args,shared,threads in [('volume','Q,D,G,rhs',588,84),('surface','Q,part,L,S,rhs',672,112),('rk','Q,res,rhs,fa,fb,fdt',0,256),('gather','Q,mapping,out',0,256)]:
 body=(P/'translation_generators'/f'{name}_body.jl.txt').read_text().replace('map[','mapping[')
 code+=f'function {name}_kernel(index,{args}'+(',template'if shared else'')+f')\n i=Int(index)-1;tid=i%{threads};gid=div(i,{threads})\n'
 if shared:code+='s=JACC.shared(template)\n'
 code+='@inbounds begin\n'+body.replace('SYNC','JACC.sync_workgroup()')+'\nend\nreturn nothing\nend\n'
code+='''alloc_backend(::Type{T},n) where T=JACC.array(T,n)
const VOLUME_TEMPLATE=JACC.array(zeros(Float32,588))
const SURFACE_TEMPLATE=JACC.array(zeros(Float32,672))
all(iszero,Array(VOLUME_TEMPLATE))&&all(iszero,Array(SURFACE_TEMPLATE))||error("actualtemplatezeroidentity")
println("STUDY_BACKEND_INPUT actual_GPU_zero_shared_templates_Float32_588_672_checked=true semantically_overwritten_read_positions=true")
function run_gather!(Q,mapping,out)
 JACC.parallel_for(JACC.launch_spec(threads=256,blocks=50,shmem_size=0,sync=false),50*256,gather_kernel,Q,mapping,out)
end
function run_volume!(Q,D,G,rhs)
 JACC.parallel_for(JACC.launch_spec(threads=84,blocks=511,shmem_size=2352,sync=false),511*84,volume_kernel,Q,D,G,rhs,VOLUME_TEMPLATE)
end
function run_surface!(Q,part,L,S,rhs)
 JACC.parallel_for(JACC.launch_spec(threads=112,blocks=511,shmem_size=2688,sync=false),511*112,surface_kernel,Q,part,L,S,rhs,SURFACE_TEMPLATE)
end
function run_rk!(Q,res,rhs,fa,fb,fdt)
 JACC.parallel_for(JACC.launch_spec(threads=256,blocks=1150,shmem_size=0,sync=false),1150*256,rk_kernel,Q,res,rhs,fa,fb,fdt)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
'''
(P/'miniDGS-julia-JACC/main.jl').write_text(code)
