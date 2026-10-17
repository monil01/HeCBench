include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
run!(pages,rank,maps,degree,n,history)=(@cuda threads=256 blocks=2 map_kernel!(pages,rank,maps,degree,n);@cuda threads=256 blocks=2 reduce_kernel!(rank,maps,n,history);nothing)
study(run!)
