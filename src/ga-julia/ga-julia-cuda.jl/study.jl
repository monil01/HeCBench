using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,first(split(source,"exit(main(ARGS))")),joinpath(@__DIR__,"main.jl"))
launch_julia!(t,q,o,n,ql,cl,th,pos)=(@cuda threads=256 blocks=4 ga_kernel!(t,q,o,n,ql,cl,th,pos);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"));ga_study(launch_julia!)
