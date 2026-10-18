using CUDA,JACC
include(joinpath(@__DIR__,"..","warp_network.jl"))
function collision_flags!(i,a,out)
 lane=Int32(i)-Int32(1)
 out[lane+Int32(1)]=collision_flag(a[lane+Int32(1)],lane)
 return
end
function collision_masks!(i,a,out)
 lane=Int32(i)-Int32(1);mask=collision_mask(a[lane+Int32(1)],lane)
 if lane==0;out[1]=mask;end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const COLLISION_SPEC=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=32,blocks=1,shmem_size=0,sync=false)
function launch_collision!(a,f,m)
 JACC.parallel_for(collision_flags!,COLLISION_SPEC,32,a,f)
 JACC.parallel_for(collision_masks!,COLLISION_SPEC,32,a,m)
end
collision_study(launch_collision!)
