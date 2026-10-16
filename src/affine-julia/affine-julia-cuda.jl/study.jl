using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"\nmain()\n"=>"\n"),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(input,output)=()->(@cuda threads=(16,16) blocks=(32,32) affine_kernel!(input,output))
affine_study(factory,affine_reference)
