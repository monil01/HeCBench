using CUDA,StaticArrays
include(joinpath(@__DIR__,"..","study_device.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,dt,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18)
 wsm_body!(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,dt,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,UInt32(blockIdx().x-1),UInt32(blockIdx().y-1),UInt32(blockDim().x),UInt32(blockDim().y),UInt32(threadIdx().x-1),UInt32(threadIdx().y-1))
 return
end
function launch(c)
 CUDA.@cuda threads=(8,8) blocks=(cld(c.nx,8),cld(c.ny,8)) kernel(c.d...,c.dt,c.bounds...)
end
study_main(CuArray,launch)
