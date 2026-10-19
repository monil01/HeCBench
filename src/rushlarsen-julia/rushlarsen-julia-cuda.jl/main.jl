using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
function rush_kernel!(s,t,dt,p,n)
 i=Int32((blockIdx().x-UInt32(1))*blockDim().x+threadIdx().x-UInt32(1));rush_body!(s,t,dt,p,n,i);return
end
function launch(c,t)
 CUDA.@cuda threads=256 blocks=cld(Int(c.n),256) rush_kernel!(c.ds,t,c.dt,c.dp,c.n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rl_study(CuArray,launch)
