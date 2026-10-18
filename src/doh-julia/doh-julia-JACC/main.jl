using CUDA,JACC
include(joinpath(@__DIR__,"..","hessian_math.jl"))
function doh_kernel!(i,a,h,w,sigma,o)
 doh_cell!(a,h,w,sigma,o,Int32(i)-Int32(1))
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const DOH_SPECS=Dict(n=>JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false) for n in (527,512,297,19))
doh_study((a,h,w,sigma,o,n)->JACC.parallel_for(doh_kernel!,DOH_SPECS[n],cld(n,256)*256,a,h,w,sigma,o))
