include("main.jl")
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(n,r,a,x)
 @cuda threads=256 blocks=cld(Int(n),256) gemv_column_kernel!(r,a,x,Int32(n))
 nothing
end
streamum_study(launch!)
