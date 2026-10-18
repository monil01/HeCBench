using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function rotate_ka!(d,p,n::Int32,width)
 i=Int32(@index(Global,Linear))
 if i<=n;@inbounds rotate_point!(d,i,p[5],p[6],p[7],p[8],width);end
end
function factory(d,p,width)
 k=rotate_ka!(CUDABackend(),256)
 n->k(d,p,n,width;ndrange=n)
end
rodrigues_study(factory)
