using CUDA,JACC
framework_array(host)=JACC.array(host)
include(joinpath(@__DIR__,"..","study_common.jl"))
hpl_study()
