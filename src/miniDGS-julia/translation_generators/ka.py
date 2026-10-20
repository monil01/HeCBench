from pathlib import Path
P=Path(__file__).resolve().parents[1];code='using CUDA, KernelAbstractions\n'
for name,args,shared in [('volume','Q,D,G,rhs',588),('surface','Q,part,L,S,rhs',672),('rk','Q,res,rhs,fa,fb,fdt',0),('gather','Q,mapping,out',0)]:
 body=(P/'translation_generators'/f'{name}_body.jl.txt').read_text().replace('map[','mapping[')
 code+=f'@kernel function {name}_kernel({args})\n tid=Int(@index(Local,Linear))-1;gid=Int(@index(Group,Linear))-1;i=Int(@index(Global,Linear))-1\n'
 if shared:code+=f's=@localmem Float32 ({shared},)\n'
 code+='@inbounds begin\n'+body.replace('SYNC','@synchronize()')+'\nend\nend\n'
code+='''alloc_backend(::Type{T},n) where T=KernelAbstractions.allocate(CUDABackend(),T,(n,))
run_gather!(Q,mapping,out)=gather_kernel(CUDABackend(),256)(Q,mapping,out;ndrange=50*256)
run_volume!(Q,D,G,rhs)=volume_kernel(CUDABackend(),84)(Q,D,G,rhs;ndrange=511*84)
run_surface!(Q,part,L,S,rhs)=surface_kernel(CUDABackend(),112)(Q,part,L,S,rhs;ndrange=511*112)
run_rk!(Q,res,rhs,fa,fb,fdt)=rk_kernel(CUDABackend(),256)(Q,res,rhs,fa,fb,fdt;ndrange=1150*256)
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
'''
(P/'miniDGS-julia-KA/main.jl').write_text(code)
