using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(out,atoms,L)
 i=@index(Global,Linear)
 @inbounds out[i]=factor(sample(i-1,L)...,atoms,L,Val(false))
end
run!(out,atoms,L)=kernel!(get_backend(out),256)(out,atoms,L;ndrange=4096)
study(run!)
