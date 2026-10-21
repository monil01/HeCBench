include(joinpath(@__DIR__,"..","study_guard.jl"))
using CUDA,Libdl
using KernelAbstractions
allocate(n)=KernelAbstractions.allocate(CUDA.CUDABackend(),Int32,(n,))
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate)
