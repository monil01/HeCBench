include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
launch!(d,n)=(@cuda threads=256 blocks=cld(n,256) wyllie_study!(d,n))
study_main(launch!)
