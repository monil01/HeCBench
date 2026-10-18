include("main.jl")
function study_launch!(d,seed)
 @cuda threads=256 blocks=257 rng_wallace_kernel!(d[1],d[2],d[3],seed)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main("cuda_jl",study_launch!)
