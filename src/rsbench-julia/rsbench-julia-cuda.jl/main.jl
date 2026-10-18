using CUDA
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
function lookup_device(args...)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 rs_store(i,args...)
 return
end
launch(args...)=(@cuda threads=256 blocks=cld(args[10],256) lookup_device(args...);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
study(CuArray,launch)
