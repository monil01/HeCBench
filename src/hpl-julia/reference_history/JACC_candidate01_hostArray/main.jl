using CUDA,JACC
framework_array(host)=JACC.Array(host)
include(joinpath(@__DIR__,"..","study_common.jl"))
hpl_study()
