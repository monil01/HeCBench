using CUDA
include(joinpath(@__DIR__,"..","warp_network.jl"))
function collision_flags!(a,out)
 lane=Int32(threadIdx().x)-Int32(1);out[lane+Int32(1)]=collision_flag(a[lane+Int32(1)],lane)
 return
end
function collision_masks!(a,out)
 lane=Int32(threadIdx().x)-Int32(1);mask=collision_mask(a[lane+Int32(1)],lane)
 if lane==0;out[1]=mask;end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch_collision!(a,f,m)
 @cuda threads=32 blocks=1 collision_flags!(a,f)
 @cuda threads=32 blocks=1 collision_masks!(a,m)
end
collision_study(launch_collision!)
