using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
source=source[findfirst("function kquantize!",source).start:findfirst("function main(",source).start-1]
include(joinpath(@__DIR__,"..","study_common.jl"));include_string(Main,source,joinpath(@__DIR__,"main.jl"))
factory(code,input,output)=()->(@cuda threads=256 blocks=1024 kquantize!(code,input,output,Int32(262144)))
quantbnb_study(factory)
