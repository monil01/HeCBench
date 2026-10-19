using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
# Hybrid CUDA-specific CAS,shuffles,staticshared andbarriers are explicit in shared helpers.
for (name,body) in [(:init_kernel!,:cc_init),(:low_kernel!,:cc_compute1),(:medium_kernel!,:cc_compute2),(:high_kernel!,:cc_compute3),(:flatten_kernel!,:cc_flatten)]
 @eval function $name(i,n,idx,adj,stat,wl,ctr)
  $body(Int32(i)-Int32(1),Int32(640*256),n,idx,adj,stat,wl,ctr)
  nothing
 end
end
function observer!(i,n,e,blocks,out)
 @inbounds out[1]=n;out[2]=e;out[3]=blocks
 nothing
end
const STAGES=(init_kernel!,low_kernel!,medium_kernel!,high_kernel!,flatten_kernel!)
const SPEC=JACC.launch_spec(threads=256,blocks=640,shmem_size=0,sync=false)
const OBSERVE=JACC.launch_spec(threads=1,blocks=1,shmem_size=0,sync=false)
function pipeline(x)
 for stage in STAGES
  JACC.parallel_for(SPEC,640*256,stage,x.c.n,x.idx,x.adj,x.stat,x.wl,x.ctr)
 end
end
cc_study(pipeline,x->JACC.parallel_for(OBSERVE,1,observer!,x.c.n,x.c.e,Int32(x.c.blocks),x.params);allocate=JACC.array)
