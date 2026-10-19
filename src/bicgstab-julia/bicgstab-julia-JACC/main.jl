# Zero authored device kernels: backend-specific vendor integration and Julia
# host solver control. JACC does not provide a portable BiCGStab kernel here.
using CUDA,JACC
JACC.@init_backend
JACC.synchronize()
include(joinpath(@__DIR__,"..","study_common.jl"))
bicg_study()
