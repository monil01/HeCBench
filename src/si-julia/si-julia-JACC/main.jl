using CUDA,Libdl,SHA,JACC
JACC.@init_backend
# Actual public JACC undef allocation avoids introducing a zero-fill/H2D phase.
# Source GPU work is vendor SGEMM; no dummy parallel_for/kernel is invented.
si_allocate(n)=JACC.array(Float32,(n,))
si_release(a)=CUDA.unsafe_free!(a) # Explicit CUDA host-level vendor/runtime interop.
include(joinpath(@__DIR__,"..","study_common.jl"))
si_study()
