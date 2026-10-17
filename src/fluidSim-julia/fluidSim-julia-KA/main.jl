using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function lbm_ka!(a,oa,b,ob,c,oc,types,w,omega)
 ij=@index(Global,NTuple)
 lbm_body!(Int32(ij[1])-Int32(1),Int32(ij[2])-Int32(1),a,oa,b,ob,c,oc,types,w,omega)
end
const BACKEND=CUDABackend()
function run!(a,oa,b,ob,c,oc,types,w,omega)
 lbm_ka!(BACKEND,(1,HEIGHT))(a,oa,b,ob,c,oc,types,w,omega;ndrange=(WIDTH,HEIGHT))
 nothing
end
study(run!)
