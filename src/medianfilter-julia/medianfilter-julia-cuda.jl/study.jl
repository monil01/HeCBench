include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_prepare(input,output,W,H)
    ()->(@cuda threads=(16,4) blocks=(cld(W,16),cld(H,4)) median_kernel!(input,output,Int32(W),Int32(H)))
end
exit(study_main(study_prepare))
