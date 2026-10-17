include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(a,f,pitched)
 w,h,d=Int32.(f[1:3]);stride=Int64(f[5])
 if f[4]
  @cuda threads=(16,8,4) blocks=(cld(w,16),cld(h,8),cld(d,4)) sigmoid3d!(Val(pitched),a,w,h,d,stride)
 else
  @cuda threads=(16,16) blocks=(cld(w,16),cld(h,16)) sigmoid2d!(Val(pitched),a,w,h,stride)
 end
end
study_main(launch!)
