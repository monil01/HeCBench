using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"exit(main(ARGS))"=>""),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
factory(a,b,out,n)=()->(@cuda threads=256 blocks=cld(n,256) cross3_kernel!(Int32(n),out,a,b))
cross_study(factory)
