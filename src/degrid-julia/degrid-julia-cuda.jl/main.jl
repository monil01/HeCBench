include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
launch(out,input,img,gcf)=@cuda threads=(32,8) blocks=2 degrid_cuda!(out,input,img,gcf)
study(launch,(T,n)->CuArray{T}(undef,n))
