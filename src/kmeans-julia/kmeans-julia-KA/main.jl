using CUDA,KernelAbstractions
@kernel function transpose_kernel!(sw,x,f,n)
 i=@index(Global,Linear)
 if i<=n
  @inbounds for a in Int32(1):f;sw[(a-1)*n+i]=x[(i-1)*f+a];end
 end
 nothing
end
@kernel function membership_kernel!(sw,c,m,k,f,n)
 i=@index(Global,Linear)
 if i<=n
  index=Int32(0);best=typemax(Float32)
  @inbounds for j in Int32(0):k-1
   dist=0f0
   for a in Int32(0):f-1;d=sw[a*n+i]-c[j*f+a+1];dist+=d*d;end
   if dist<best;best=dist;index=j;end
  end
  @inbounds m[i]=index
 end
 nothing
end
const BACKEND=CUDABackend()
first(c)=transpose_kernel!(BACKEND,256)(c.sw,c.x,Int32(c.f),Int32(c.n);ndrange=div(c.n+256,256)*256)
second(c,k)=membership_kernel!(BACKEND,256)(c.sw,c.cent,c.mem,Int32(k),Int32(c.f),Int32(c.n);ndrange=div(c.n+256,256)*256)
function alloc(v)
 a=KernelAbstractions.allocate(BACKEND,eltype(v),(length(v),));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(alloc,first,second)
