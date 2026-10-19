using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
for (name,body)in [(:forward!,:bray_forward),(:components!,:bray_components),(:grad1!,:bray_grad1),(:grad2!,:bray_grad2)]
 @eval function $name(i,stride,c,a,b,g,o,sd,ss,g1,g2)
  $body(Int32(i)-Int32(1),stride,c,a,b,g,o,sd,ss,g1,g2)
  nothing
 end
end
function observer!(i,c,p)
 @inbounds p[1]=c.b;p[2]=c.m;p[3]=c.n;p[4]=c.k;p[5]=c.repeats
 nothing
end
const BRAY_KERNELS=(forward!,components!,grad1!,grad2!)
const BRAY_SPECS=Dict(n=>JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false) for n in unique([n for c in BRAY_CASES for n in (c.nq,c.na,c.nb)]))
const OBSERVE=JACC.launch_spec(threads=1,blocks=1,shmem_size=0,sync=false)
function launch(stage,x)
 count=stage<=2 ? x.c.nq : stage==3 ? x.c.na : x.c.nb
 stride=Int32(cld(count,256)*256)
 JACC.parallel_for(BRAY_SPECS[count],Int(stride),BRAY_KERNELS[stage],stride,x.args...)
end
bray_study(launch,x->JACC.parallel_for(OBSERVE,1,observer!,x.c,x.params);allocate=JACC.array)
