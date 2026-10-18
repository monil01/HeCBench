using JACC
import Atomix
include("../study_common.jl")
const TEMPLATE=CUDA.zeros(Int32,1)
function rsc_jacc(i,p,f,n,m,e,conv,id,candidates,counts,nt::Int32,stride::Int32,template)
 tx=rem(Int32(i)-Int32(1),nt);model=div(Int32(i)-Int32(1),nt)
 shared=JACC.shared(template)
 while model<m
  if tx==0;shared[1]=0;end
  JACC.sync_workgroup()
  base=4*model
  if p[base+1]!=-2011
   local_count=Int32(0);j=tx
   while j<n
    fv=f[j+1]
    rx=fv.x+unsafe_trunc(Int32,(Float32(fv.x)-p[base+1])*p[base+3])-unsafe_trunc(Int32,(Float32(fv.y)-p[base+2])*p[base+4])-fv.vx
    ry=fv.y+unsafe_trunc(Int32,(Float32(fv.y)-p[base+2])*p[base+3])+unsafe_trunc(Int32,(Float32(fv.x)-p[base+1])*p[base+4])-fv.vy
    local_count+=Int32(abs(rx)>=e||abs(ry)>=e);j+=nt
   end
   Atomix.@atomic :monotonic shared[1] += local_count
   JACC.sync_workgroup()
   if tx==0&&shared[1]<Float32(n)*conv
    index=(Atomix.@atomic :monotonic id[1] + Int32(1))[1]+1;candidates[index]=model;counts[index]=shared[1]
   end
  end
  model+=stride
 end
 nothing
end
const SPECS=Dict((b,t)=>JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=t,blocks=b,shmem_size=4,sync=false) for (b,t) in [(17,64),(3,32),(11,64),(19,64)])
function launch(c)
 JACC.parallel_for(rsc_jacc,SPECS[(c.blocks,c.threads)],c.blocks*c.threads,c.p.data,c.gflows,c.n,c.m,c.e,c.conv,c.id.data,c.c.data,c.o.data,Int32(c.threads),Int32(c.blocks),TEMPLATE)
end
rsc_study(launch,"jacc")
