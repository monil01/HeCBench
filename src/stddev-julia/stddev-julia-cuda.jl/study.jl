using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,first(split(source,"function reference_value")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(output,input)=()->stddev_gpu!(output,input,Int32(256),Int32(4096),true)
stddev_study(factory)
