include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
launch(input,output,constants)=@cuda threads=64 blocks=3 tree_cuda!(input,output,constants)
study(launch,(T,n)->CuArray{T}(undef,n))
