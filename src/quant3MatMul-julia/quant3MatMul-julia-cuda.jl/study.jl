using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,first(split(source,"function main(")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(vec,mat,out,scales,zeros)=()->(@cuda threads=256 blocks=(16,4) quant3_kernel!(vec,mat,out,scales,zeros,Int32(1024)))
quant3_study(factory)
