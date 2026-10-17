# Native behavior study only; known scientific defect. Original genuine CCD is historical_ccd_main.jl/legacy_main.jl.
using CUDA
include(joinpath(@__DIR__,"..","frozen_pose_math.jl"))
function native_behavior_kernel!(x,y,a,n)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if i<=n
  z=frozen_pose_angles(x[i],y[i]);@inbounds a[3i-2]=z[1];a[3i-1]=z[2];a[3i]=z[3]
 end
 return
end
