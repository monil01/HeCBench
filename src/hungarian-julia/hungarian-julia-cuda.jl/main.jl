using CUDA
const H_MODEL=:cuda_jl
include(joinpath(@__DIR__,"..","hungarian_math.jl"))
function hungarian_kernel!(v,S)
 hphase!(v,Int32(threadIdx().x-1),Int32(blockIdx().x-1),Int32(blockDim().x),Int32(gridDim().x),S)
 return
end
function launch!(v,S,nb,nt,shared)
 CUDA.@cuda threads=nt blocks=nb shmem=shared hungarian_kernel!(v,S)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
hungarian_study(launch!)
