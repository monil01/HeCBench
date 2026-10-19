using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
for (name,body) in [(:init_kernel!,:cc_init),(:low_kernel!,:cc_compute1),(:medium_kernel!,:cc_compute2),(:high_kernel!,:cc_compute3),(:flatten_kernel!,:cc_flatten)]
 @eval function $name(n,idx,adj,stat,wl,ctr)
  from=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);incr=Int32(gridDim().x*blockDim().x)
  $body(from,incr,n,idx,adj,stat,wl,ctr)
  nothing
 end
end
function observer!(n,e,blocks,out)
 @inbounds out[1]=n;out[2]=e;out[3]=blocks
 return
end
function pipeline(x)
 c=x.c
 CUDA.@cuda threads=256 blocks=c.blocks init_kernel!(c.n,x.idx,x.adj,x.stat,x.wl,x.ctr)
 CUDA.@cuda threads=256 blocks=c.blocks low_kernel!(c.n,x.idx,x.adj,x.stat,x.wl,x.ctr)
 CUDA.@cuda threads=256 blocks=c.blocks medium_kernel!(c.n,x.idx,x.adj,x.stat,x.wl,x.ctr)
 CUDA.@cuda threads=256 blocks=c.blocks high_kernel!(c.n,x.idx,x.adj,x.stat,x.wl,x.ctr)
 CUDA.@cuda threads=256 blocks=c.blocks flatten_kernel!(c.n,x.idx,x.adj,x.stat,x.wl,x.ctr)
end
cc_study(pipeline,x->CUDA.@cuda(threads=1,blocks=1,observer!(x.c.n,x.c.e,Int32(x.c.blocks),x.params)))
