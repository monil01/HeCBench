using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function ntt_ka!(y,x)
 t=Int32(@index(Local,Linear))-Int32(1);b=Int32(@index(Group,Linear))-Int32(1);buffer=@localmem UInt64(512,)
 phase1!(x,buffer,b,t);@synchronize;phase2!(y,buffer,b,t)
end
backend=KernelAbstractions.get_backend(CUDA.zeros(UInt8,1));run_study((y,x)->ntt_ka!(backend,64)(y,x;ndrange=8192),"ka")
