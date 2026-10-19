using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
function sequence_cuda!(inverse,args...)
 bsw_device!(inverse,args...)
 nothing
end
bsw_study((inverse,threads,args...)->CUDA.@cuda(threads=threads,blocks=N,shmem=SHMEM,sequence_cuda!(inverse,args...)))
