include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
convert_launch(dd,dr,dc,dv,m,n)=@cuda threads=128 blocks=cld(m,Int32(128)) convert_cuda!(dd,dr,dc,dv,m,n)
study(convert_launch,(T,n)->CuArray{T}(undef,n))
