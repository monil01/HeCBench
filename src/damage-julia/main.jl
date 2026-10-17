include(joinpath(@__DIR__,"damage-julia-cuda.jl","main.jl"))
damage_study(factory;n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 262144,repeats=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10)
