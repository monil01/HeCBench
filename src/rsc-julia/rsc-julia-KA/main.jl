using KernelAbstractions
import Atomix
include("../study_common.jl")
@kernel function rsc_ka(p,f,n,m,e,conv,id,candidates,counts,nt::Int32,stride::Int32)
 tx=Int32(@index(Local,Linear))-Int32(1);model=Int32(@index(Group,Linear))-Int32(1)
 shared=@localmem Int32 (1,)
 while model<m
  if tx==0;shared[1]=0;end
  @synchronize
  base=4*model
  if p[base+1]!=-2011
   local_count=Int32(0);i=tx
   while i<n
    fv=f[i+1]
    rx=fv.x+unsafe_trunc(Int32,(Float32(fv.x)-p[base+1])*p[base+3])-unsafe_trunc(Int32,(Float32(fv.y)-p[base+2])*p[base+4])-fv.vx
    ry=fv.y+unsafe_trunc(Int32,(Float32(fv.y)-p[base+2])*p[base+3])+unsafe_trunc(Int32,(Float32(fv.x)-p[base+1])*p[base+4])-fv.vy
    local_count+=Int32(abs(rx)>=e||abs(ry)>=e);i+=nt
   end
   Atomix.@atomic :monotonic shared[1] += local_count
   @synchronize
   if tx==0&&shared[1]<Float32(n)*conv
    j=(Atomix.@atomic :monotonic id[1] + Int32(1))[1]+1;candidates[j]=model;counts[j]=shared[1]
   end
  end
  model+=stride
 end
end
const K32=rsc_ka(CUDABackend(),32)
const K64=rsc_ka(CUDABackend(),64)
function launch(c)
 k=c.threads==32 ? K32 : K64
 k(c.p.data,c.gflows,c.n,c.m,c.e,c.conv,c.id.data,c.c.data,c.o.data,Int32(c.threads),Int32(c.blocks);ndrange=c.threads*c.blocks)
end
rsc_study(launch,"ka")
