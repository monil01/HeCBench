using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(CUDA.CUDABackend(),T,(n,)))
