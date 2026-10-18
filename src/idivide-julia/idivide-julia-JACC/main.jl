using CUDA,JACC
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
function id_capture!(i,xs,out,n::Int32,d,base::Int32)
 if i<=n
  @inbounds x=xs[i+1];q=quotient_value(x,d)
  @inbounds out[base+2+i]=q
  @inbounds out[base+2+n+i]=x-q*plain_d(d)
  if i==1;@inbounds out[base+2]=plain_d(d);end
 end
 return
end
function id_check!(index,d::FastDiv,checks,base::Int32)
 i=Int32(index)-Int32(1)
 for x in (i,-i)
  q=div(x,d.d);f=quotient_value(x,d)
  if q!=f
   old=(Atomix.@atomic :monotonic checks[base+2] + Int32(1))[1]
   if old==0;@inbounds checks[base+3]=x;checks[base+4]=q;checks[base+5]=f;end
  end
 end
 return
end
function factory(x,out,checks,n)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 launch! =(d,base)->JACC.parallel_for(spec,n,id_capture!,x,out,n,d,base)
 check! =(d,base)->JACC.parallel_for(spec,4096,id_check!,d,checks,base)
 launch!,check!
end
idivide_study(factory)
