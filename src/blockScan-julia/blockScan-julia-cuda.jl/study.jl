include("main.jl")
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_launch!(a,o,::Val{BT},::Val{IPT}) where {BT,IPT}
 @cuda threads=BT blocks=1 block_prefix_sum_kernel!(a,o,Int32(1024))
end
study_main(study_launch!)
