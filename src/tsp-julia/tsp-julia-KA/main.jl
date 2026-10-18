using CUDA, KernelAbstractions
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function twoopt!(n,x,y,g,climbs,best,::Val{T}) where T
 tid=@index(Local,Linear)
 bid=@index(Group,Linear)
 sh=@localmem Int32 (T+384,)
 twoopt_body!(Int(n),x,y,g,climbs,best,sh,Int(tid)-1,Int(bid)-1,T)
end
@kernel function pairs!(out,x,y,n)
 idx=@index(Global,Linear)
 i=Int(idx)-1
 if i<n*n
  a=i÷n;b=i%n
  @inbounds out[i+1]=distance(x[a+1],y[a+1],x[b+1],y[b+1])
 end
end
function launch(c)
 twoopt!(CUDA.CUDABackend(),c.t)(Int32(c.n),c.x,c.y,c.g,c.climbs,c.best,Val(c.t);ndrange=c.t*c.r)
end
function diagnostic(c)
 pairs!(CUDA.CUDABackend(),256)(c.pairs,c.x,c.y,Int32(c.n);ndrange=c.n*c.n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->KernelAbstractions.allocate(CUDA.CUDABackend(),Int32,(n,)),launch,diagnostic)
