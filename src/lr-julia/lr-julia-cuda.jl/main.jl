using CUDA
function regress!(d,q)
 l=Int(threadIdx().x);g=Int(blockIdx().x);i=(g-1)*63+l
 s=CuDynamicSharedArray(Float32,252)
 @inbounds begin
 x=d[2i-1];y=d[2i];s[4l-3]=x;s[4l-2]=y;s[4l-1]=x*y;s[4l]=x*x
 end
 sync_threads()
 width=31;old=63
 while width>0
  if l<=width
   @inbounds for f in 1:4
    k=4(l-1)+f;s[k]+=s[k+4width]
    if l==width&&isodd(old);s[k]+=s[4(old-1)+f];end
   end
  end
  sync_threads();old=width;width=div(width,2)
 end
 if l==1;@inbounds for f in 1:4;q[4(g-1)+f]=s[f];end;end
 return
end
function rsq!(d,z,m,a,b)
 l=Int(threadIdx().x);g=Int(blockIdx().x);i=(g-1)*63+l
 s=CuDynamicSharedArray(Float32,126)
 @inbounds begin
 x=d[2i-1];y=d[2i];v=y-m;e=x*b+a-m;s[2l-1]=v*v;s[2l]=e*e
 end
 sync_threads();width=31;old=63
 while width>0
  if l<=width
   @inbounds for f in 1:2
    k=2(l-1)+f;s[k]+=s[k+2width]
    if l==width&&isodd(old);s[k]+=s[2(old-1)+f];end
   end
  end
  sync_threads();old=width;width=div(width,2)
 end
 if l==1;@inbounds for f in 1:2;z[2(g-1)+f]=s[f];end;end
 return
end
first(c)=(@cuda threads=63 blocks=div(c.n,63) shmem=1008 regress!(c.d,c.q))
second(c,m,a,b)=(@cuda threads=63 blocks=div(c.n,63) shmem=504 rsq!(c.d,c.z,m,a,b))
alloc(v)=CuArray(v)
include(joinpath(@__DIR__,"..","study_common.jl"))
study(alloc,first,second)
