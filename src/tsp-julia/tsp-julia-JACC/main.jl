using CUDA
import JACC
JACC.@init_backend
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
const TSP_TEMPLATES=Dict(t=>JACC.zeros(Int32,t+384) for t in (98,128))
function twoopt!(idx,n,x,y,g,climbs,best,template,::Val{T}) where T
 sh=JACC.shared(template)
 tid=(Int(idx)-1)%T;bid=(Int(idx)-1)÷T
 twoopt_body!(Int(n),x,y,g,climbs,best,sh,tid,bid,T)
 nothing
end
@inline function pairs!(idx,out,x,y,n)
 i=Int(idx)-1
 if i<n*n
  a=i÷n;b=i%n
  @inbounds out[i+1]=distance(x[a+1],y[a+1],x[b+1],y[b+1])
 end
 nothing
end
function launch(c)
 spec=JACC.launch_spec(;threads=c.t,blocks=c.r,shmem_size=4*(c.t+384),sync=false)
 JACC.parallel_for(spec,c.t*c.r,twoopt!,Int32(c.n),c.x,c.y,c.g,c.climbs,c.best,TSP_TEMPLATES[c.t],Val(c.t))
end
function diagnostic(c)
 spec=JACC.launch_spec(;threads=256,blocks=cld(c.n*c.n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,c.n*c.n,pairs!,c.pairs,c.x,c.y,Int32(c.n))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->JACC.zeros(Int32,n),launch,diagnostic)
