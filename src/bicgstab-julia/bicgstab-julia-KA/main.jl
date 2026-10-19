# Source has no authored device kernels. Backend-specific vendor integration;
# the Julia host solver is translated, not a portable KA solver kernel.
using CUDA,KernelAbstractions
const TARGET_BACKEND=CUDABackend()
KernelAbstractions.synchronize(TARGET_BACKEND)
include(joinpath(@__DIR__,"..","study_common.jl"))
bicg_study()
