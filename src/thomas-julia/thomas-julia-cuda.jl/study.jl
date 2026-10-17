include("main.jl")
include("../study_common.jl")
function launch_ref(l,d,u,r,m,n)
 @cuda threads=128 blocks=(Int(n)÷128+1) thomas_kernel!(l,d,u,r,m,n)
end
thomas_study(launch_ref)
