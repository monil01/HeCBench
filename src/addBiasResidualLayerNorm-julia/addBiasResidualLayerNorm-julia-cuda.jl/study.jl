using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,first(split(source,"function reference!")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(out,input,bias,gamma,beta)=()->(@cuda threads=256 blocks=512 layernorm_kernel!(out,input,bias,gamma,beta,1f-6,Int32(256)))
layer_study(factory)
