using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n))
