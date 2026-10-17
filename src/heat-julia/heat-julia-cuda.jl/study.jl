include("main.jl")
include("../study_common.jl")
function launch_ref(dest,src,n,r,r2)
 @cuda threads=256 blocks=cld(Int(n)*Int(n),256) solve_kernel!(dest,src,n,r,r2)
end
heat_study(launch_ref)
