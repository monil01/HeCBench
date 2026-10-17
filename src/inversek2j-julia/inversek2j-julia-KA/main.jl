# Native known-defect behavior study; not a validated inverse-kinematics solver.
using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","frozen_pose_math.jl"))
@kernel function behavior_kernel!(x,y,a)
 i=@index(Global,Linear)
 if i<=length(x)
  z=frozen_pose_angles(x[i],y[i]);@inbounds a[3i-2]=z[1];a[3i-1]=z[2];a[3i]=z[3]
 end
end
function study_launch!(x,y,a)
 behavior_kernel!(KernelAbstractions.get_backend(x),128)(x,y,a;ndrange=length(x))
end
study_main(study_launch!)
