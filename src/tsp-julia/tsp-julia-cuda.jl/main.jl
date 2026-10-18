using CUDA
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
function twoopt!(n,x,y,g,climbs,best,::Val{T})where T
 tid=Int(threadIdx().x)-1;bid=Int(blockIdx().x)-1
 sh=CuDynamicSharedArray(Int32,T+384)
 twoopt_body!(Int(n),x,y,g,climbs,best,sh,tid,bid,T)
 return
end
function pairs!(out,x,y,n)
 i=(Int(blockIdx().x)-1)*Int(blockDim().x)+Int(threadIdx().x)-1
 if i<n*n
  a=i÷n;b=i%n
  @inbounds out[i+1]=distance(x[a+1],y[a+1],x[b+1],y[b+1])
 end
 return
end
function launch(c)
 @cuda threads=c.t blocks=c.r shmem=4*(c.t+384) twoopt!(Int32(c.n),c.x,c.y,c.g,c.climbs,c.best,Val(c.t))
end
function diagnostic(c)
 @cuda threads=256 blocks=cld(c.n*c.n,256) pairs!(c.pairs,c.x,c.y,Int32(c.n))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->CUDA.CuArray{Int32}(undef,n),launch,diagnostic)
