using CUDA, KernelAbstractions
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function bond_kernel(curve,repo,current,maturity,clean,bond,strike,dirty,acc,outclean,yields,n)
 g=Int32(@index(Global,Linear))-Int32(1)
 bonds_body!(curve,repo,current,maturity,clean,bond,strike,dirty,acc,outclean,yields,n,g)
end
function launch(dev,n)
 bond_kernel(KernelAbstractions.get_backend(dev[1]),256)(dev...,Int32(n);ndrange=cld(n,256)*256)
 nothing
end
function allocate(v)
 a=KernelAbstractions.allocate(CUDA.CUDABackend(),eltype(v),size(v));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launch)
