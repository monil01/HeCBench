using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,first(split(source,"function reference!")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(input,output,entropy,epochs,tol,n,k)=()->(@cuda threads=256 blocks=cld(n,256) sigmas_kernel!(input,output,entropy,epochs,tol,n,k))
perplexity_study(factory)
