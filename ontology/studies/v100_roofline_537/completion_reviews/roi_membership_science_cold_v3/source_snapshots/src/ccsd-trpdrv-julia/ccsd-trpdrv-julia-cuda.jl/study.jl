include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_launch!(arrs,eorb,ea,out,nv)
 @cuda threads=(16,16) blocks=(cld(nv,16),cld(nv,16)) ccsd_kernel!(arrs...,eorb,out,ea,Int32(0),Int32(3),Int32(nv))
end
study_main(study_launch!)
