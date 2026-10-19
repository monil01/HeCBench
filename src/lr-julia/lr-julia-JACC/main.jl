using CUDA
import JACC
JACC.@init_backend
function regress!(i,d,q,scratch)
 l=mod(i-1,63)+1;g=div(i-1,63)+1
 s=JACC.shared(scratch)
 @inbounds begin
 x=d[2i-1];y=d[2i];s[4l-3]=x;s[4l-2]=y;s[4l-1]=x*y;s[4l]=x*x
 end
 JACC.sync_workgroup()
 width=31;old=63
 while width>0
  if l<=width
   @inbounds for f in 1:4
    k=4(l-1)+f;s[k]+=s[k+4width]
    if l==width&&isodd(old);s[k]+=s[4(old-1)+f];end
   end
  end
  JACC.sync_workgroup();old=width;width=div(width,2)
 end
 if l==1;@inbounds for f in 1:4;q[4(g-1)+f]=s[f];end;end
 return
end
function rsq!(i,d,z,m,a,b,scratch)
 l=mod(i-1,63)+1;g=div(i-1,63)+1
 s=JACC.shared(scratch)
 @inbounds begin
 x=d[2i-1];y=d[2i];v=y-m;e=x*b+a-m;s[2l-1]=v*v;s[2l]=e*e
 end
 JACC.sync_workgroup();width=31;old=63
 while width>0
  if l<=width
   @inbounds for f in 1:2
    k=2(l-1)+f;s[k]+=s[k+2width]
    if l==width&&isodd(old);s[k]+=s[2(old-1)+f];end
   end
  end
  JACC.sync_workgroup();old=width;width=div(width,2)
 end
 if l==1;@inbounds for f in 1:2;z[2(g-1)+f]=s[f];end;end
 return
end
# Pinned JACC.shared copies the whole scratch vector before our lane initialization.
# This extra staging/barrier is an explicit framework mapping difference.
const SCRATCH4=JACC.zeros(Float32,252)
const SCRATCH2=JACC.zeros(Float32,126)
first(c)=JACC.parallel_for(JACC.launch_spec(threads=63,blocks=div(c.n,63),shmem_size=1008,sync=false),c.n,regress!,c.d,c.q,SCRATCH4)
second(c,m,a,b)=JACC.parallel_for(JACC.launch_spec(threads=63,blocks=div(c.n,63),shmem_size=504,sync=false),c.n,rsq!,c.d,c.z,m,a,b,SCRATCH2)
function alloc(v)
 a=JACC.zeros(Float32,length(v));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(alloc,first,second)
