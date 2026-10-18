using CUDA
include(joinpath(@__DIR__,"..","lbm_math.jl"))
function lbm_kernel!(a,b,v,f,d,w,n,omega,lid)
 x=Int32(threadIdx().x-1)+Int32(blockIdx().x-1)*Int32(64);y=Int32(blockIdx().y-1);z=Int32(blockIdx().z-1)
 lbm_cell!(a,b,v,f,d,w,x,y,z,n,omega,lid)
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
lbm_study((a,b,v,f,d,w,n,omega,lid)->(@cuda threads=64 blocks=(1,17,17) lbm_kernel!(a,b,v,f,d,w,n,omega,lid)))
