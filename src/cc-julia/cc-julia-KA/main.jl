using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
# Hybrid backend-specific components are explicit: CUDA CAS/shuffle/staticshared/barriers.
for (name,body) in [(:init_kernel!,:cc_init),(:low_kernel!,:cc_compute1),(:medium_kernel!,:cc_compute2),(:high_kernel!,:cc_compute3),(:flatten_kernel!,:cc_flatten)]
 @eval @kernel function $name(n,idx,adj,stat,wl,ctr)
  from=Int32(@index(Global,Linear))-Int32(1);incr=Int32(640*256)
  $body(from,incr,n,idx,adj,stat,wl,ctr)
 end
end
@kernel function observer!(n,e,blocks,out)
 @inbounds out[1]=n;out[2]=e;out[3]=blocks
end
const STAGES=(init_kernel!(CUDABackend(),256),low_kernel!(CUDABackend(),256),medium_kernel!(CUDABackend(),256),high_kernel!(CUDABackend(),256),flatten_kernel!(CUDABackend(),256))
const OBSERVE=observer!(CUDABackend(),1)
function pipeline(x)
 for stage in STAGES
  stage(x.c.n,x.idx,x.adj,x.stat,x.wl,x.ctr;ndrange=x.c.blocks*256)
 end
end
cc_study(pipeline,x->OBSERVE(x.c.n,x.c.e,Int32(x.c.blocks),x.params;ndrange=1))
