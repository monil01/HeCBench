include(joinpath(@__DIR__,"..","study_guard.jl"))
using CUDA
function remote_double!(source,target)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 @inbounds target[i+1]=source[i+1]*2f0
 return
end
allocate(n)=CUDA.zeros(Float32,n)
launch(n,source,target)=(@cuda threads=256 blocks=n÷256 remote_double!(source,target);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launch)
