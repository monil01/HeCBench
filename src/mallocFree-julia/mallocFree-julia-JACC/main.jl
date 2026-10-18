using CUDA,JACC
JACC.@init_backend
# No device kernel in source: explicit CUDA-runtime allocation/free interoperability.
# No parallel_for callback, framework pool substitution, or numerical output is claimed.
include(joinpath(@__DIR__,"..","study_common.jl"))
mallocfree_study()
