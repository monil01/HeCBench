using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
for (name,body)in [(:forward!,:bray_forward),(:components!,:bray_components),(:grad1!,:bray_grad1),(:grad2!,:bray_grad2)]
 @eval @kernel function $name(stride,c,a,b,g,o,sd,ss,g1,g2)
  i=Int32(@index(Global,Linear))-Int32(1)
  $body(i,stride,c,a,b,g,o,sd,ss,g1,g2)
 end
end
@kernel function observer!(c,p)
 @inbounds p[1]=c.b;p[2]=c.m;p[3]=c.n;p[4]=c.k;p[5]=c.repeats
end
const BRAY_KERNELS=(forward!(CUDABackend(),256),components!(CUDABackend(),256),grad1!(CUDABackend(),256),grad2!(CUDABackend(),256))
const OBSERVE=observer!(CUDABackend(),1)
function launch(stage,x)
 count=stage<=2 ? x.c.nq : stage==3 ? x.c.na : x.c.nb
 stride=Int32(cld(count,256)*256)
 BRAY_KERNELS[stage](stride,x.args...;ndrange=Int(stride))
end
bray_study(launch,x->OBSERVE(x.c,x.params;ndrange=1))
