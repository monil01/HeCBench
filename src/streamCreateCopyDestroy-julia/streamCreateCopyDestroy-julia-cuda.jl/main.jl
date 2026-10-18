using CUDA
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->CUDA.CuArray{Float32}(undef,n))
