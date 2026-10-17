using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,first(split(source,"function main(")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(input,ids,values)=()->(@cuda threads=256 blocks=2048 topk_permutation_kernel!(input,ids,values,Int32(128),Int32(4096),Int32(1024),Int32(524288)))
topk_study(factory)
