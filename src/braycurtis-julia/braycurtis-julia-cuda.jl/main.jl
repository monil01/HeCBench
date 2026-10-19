using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
for (name,body)in [(:forward!,:bray_forward),(:components!,:bray_components),(:grad1!,:bray_grad1),(:grad2!,:bray_grad2)]
 @eval function $name(c,a,b,g,o,sd,ss,g1,g2)
  i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);stride=Int32(gridDim().x*blockDim().x)
  $body(i,stride,c,a,b,g,o,sd,ss,g1,g2)
  nothing
 end
end
function observer!(c,p)
 @inbounds p[1]=c.b;p[2]=c.m;p[3]=c.n;p[4]=c.k;p[5]=c.repeats
 return
end
const BRAY_KERNELS=(forward!,components!,grad1!,grad2!)
function launch(stage,x)
 count=stage<=2 ? x.c.nq : stage==3 ? x.c.na : x.c.nb
 CUDA.@cuda threads=256 blocks=cld(count,256) BRAY_KERNELS[stage](x.args...)
end
bray_study(launch,x->CUDA.@cuda(threads=1,blocks=1,observer!(x.c,x.params)))
