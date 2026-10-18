using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
function rng_kernel!(out,n::UInt64,seed::UInt32)
 tid=UInt32(blockDim().x)*(UInt32(blockIdx().x)-UInt32(1))+UInt32(threadIdx().x)-UInt32(1)
 kiss_body!(out,n,seed,tid,UInt64(1048576))
 return
end
function launch(out,n,seed)
 @cuda threads=256 blocks=4096 rng_kernel!(out,n,seed)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
