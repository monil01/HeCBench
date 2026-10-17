ENV["LAPLACE_STUDY_NUM"]="256"
include("main.jl")
include("../study_common.jl")
function launch_reference(color,a,r,b,norm)
 if color==1
  @cuda threads=(128,2) blocks=(1,128) red_kernel!(a...,b,r,norm)
 else
  @cuda threads=(128,2) blocks=(1,128) black_kernel!(a...,r,b,norm)
 end
end
laplace_study(launch_reference)
