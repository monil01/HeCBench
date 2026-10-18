using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","hessian_math.jl"))
@kernel unsafe_indices=true function doh_kernel!(a,h,w,sigma,o)
 i=Int32(@index(Global,Linear))-Int32(1)
 doh_cell!(a,h,w,sigma,o,i)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const DOH_KA=doh_kernel!(CUDABackend(),256)
doh_study((a,h,w,sigma,o,n)->DOH_KA(a,h,w,sigma,o;ndrange=cld(n,256)*256))
