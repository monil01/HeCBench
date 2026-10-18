using CUDA
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
function bond_kernel(args...)
 g=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 bonds_body!(args...,g)
 nothing
end
function launch(dev,n)
 @cuda threads=256 blocks=cld(n,256) bond_kernel(dev...,Int32(n))
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(CuArray,launch)
