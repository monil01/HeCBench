using CUDA
framework_array(host) = CuArray(host)
include(joinpath(@__DIR__,"..","study_common.jl"))
hpl_study()
