using CUDA
import JACC
JACC.@init_backend
CUDA.allowscalar(false)
# No authored device kernels: source performs CUDA runtime copies/lifecycles.
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->JACC.zeros(Float32,n))
