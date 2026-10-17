include("main.jl")
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(a,p,w,i,r,k)
 @cuda threads=256 blocks=257 softmax_cuda!(a,p)
 @cuda threads=256 blocks=257 topk_cuda!(p,w,i,r,Int32(k))
 nothing
end
moe_study(launch!)
