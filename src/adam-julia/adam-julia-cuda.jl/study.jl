using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,first(split(source,"function reference_cpu!")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(p,m,v,g,n,steps)=()->(@cuda threads=256 blocks=cld(n,256) adam_kernel!(p,m,v,g,0.9f0,0.999f0,1f-8,256f0,1f-3,steps,n,0.5f0))
adam_study(factory)
