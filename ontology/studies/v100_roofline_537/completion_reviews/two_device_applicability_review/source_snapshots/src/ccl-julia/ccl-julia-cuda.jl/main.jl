using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
ccl_execute(x->CuArray(x))
