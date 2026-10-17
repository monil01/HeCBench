# Known-defect native executable behavior only; not validated inverse kinematics.
using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","frozen_pose_math.jl"))
@inline function behavior_callback!(i,x,y,a)
 z=frozen_pose_angles(x[i],y[i]);@inbounds a[3i-2]=z[1];a[3i-1]=z[2];a[3i]=z[3]
end
function study_launch!(x,y,a)
 backend=JACC.get_backend(:cuda)
 spec=JACC.LaunchSpec{typeof(backend)}(;threads=128,blocks=cld(length(x),128),shmem_size=0,sync=false)
 JACC.parallel_for(behavior_callback!,spec,length(x),x,y,a)
end
study_main(study_launch!)
