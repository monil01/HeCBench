include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_launch!(x,y,a)
 @cuda threads=128 blocks=cld(length(x),128) native_behavior_kernel!(x,y,a,length(x))
end
study_main(study_launch!)
