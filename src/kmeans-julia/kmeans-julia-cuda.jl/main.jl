using CUDA
function transpose_kernel!(sw,x,f,n)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if i<=n
  @inbounds for a in Int32(1):f;sw[(a-1)*n+i]=x[(i-1)*f+a];end
 end
 return
end
function membership_kernel!(sw,c,m,k,f,n)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if i<=n
  index=Int32(0);best=typemax(Float32)
  @inbounds for j in Int32(0):k-1
   dist=0f0
   for a in Int32(0):f-1;d=sw[a*n+i]-c[j*f+a+1];dist+=d*d;end
   if dist<best;best=dist;index=j;end
  end
  @inbounds m[i]=index
 end
 return
end
first(c)=(@cuda threads=256 blocks=div(c.n+256,256) transpose_kernel!(c.sw,c.x,Int32(c.f),Int32(c.n)))
second(c,k)=(@cuda threads=256 blocks=div(c.n+256,256) membership_kernel!(c.sw,c.cent,c.mem,Int32(k),Int32(c.f),Int32(c.n)))
alloc(v)=CuArray(v)
include(joinpath(@__DIR__,"..","study_common.jl"))
study(alloc,first,second)
