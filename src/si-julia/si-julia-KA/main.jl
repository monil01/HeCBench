using CUDA,Libdl,SHA,KernelAbstractions
# Source GPU computation is vendor SGEMM: zero translated @kernel bodies.
# KA's genuine backend allocation returns a CuArray. Explicit CUDA host-level
# release is required: generic KA.unsafe_free! is an optional default no-op.
si_allocate(n)=KernelAbstractions.allocate(CUDA.CUDABackend(),Float32,(n,))
si_release(a)=CUDA.unsafe_free!(a)
include(joinpath(@__DIR__,"..","study_common.jl"))
si_study()
