using CUDA, KernelAbstractions
CUDA.allowscalar(false)
# No authored device kernels: native benchmark is runtime copy/stream lifecycle.
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->KernelAbstractions.allocate(CUDA.CUDABackend(),Float32,(n,)))
