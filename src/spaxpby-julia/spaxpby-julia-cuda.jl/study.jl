include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_launch!(y,idx,x,a,b)
 n=Int32(length(y));k=Int32(length(x));@cuda threads=256 blocks=cld(n,256) scale_kernel!(y,b,n)
 @cuda threads=256 blocks=cld(k,256) axpby_kernel!(y,idx,x,a,k)
end
study_main(study_launch!)
