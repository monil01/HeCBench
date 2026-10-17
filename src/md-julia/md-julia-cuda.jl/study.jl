include("main.jl")
include("../study_common.jl")
function launch_reference(p,f,neighbors,n)
 @cuda threads=256 blocks=cld(Int(n),256) md_kernel!(p...,f...,neighbors,n,Int32(128),1.5f0,2f0,13.5f0)
end
md_study(launch_reference)
