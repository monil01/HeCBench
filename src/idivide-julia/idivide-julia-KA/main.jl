using CUDA,KernelAbstractions
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function id_capture!(xs,out,n::Int32,d,base::Int32)
 i=@index(Global,Linear)
 if i<=n
  @inbounds x=xs[i+1];q=quotient_value(x,d)
  @inbounds out[base+2+i]=q
  @inbounds out[base+2+n+i]=x-q*plain_d(d)
  if i==1;@inbounds out[base+2]=plain_d(d);end
 end
end
@kernel function id_check!(d::FastDiv,checks,base::Int32)
 i=Int32(@index(Global,Linear))-Int32(1)
 for x in (i,-i)
  q=div(x,d.d);f=quotient_value(x,d)
  if q!=f
   old=(Atomix.@atomic :monotonic checks[base+2] + Int32(1))[1]
   if old==0;@inbounds checks[base+3]=x;checks[base+4]=q;checks[base+5]=f;end
  end
 end
end
function factory(x,out,checks,n)
 k=id_capture!(CUDABackend(),256);c=id_check!(CUDABackend(),256)
 launch! =(d,base)->k(x,out,n,d,base;ndrange=n)
 check! =(d,base)->c(d,checks,base;ndrange=4096)
 launch!,check!
end
idivide_study(factory)
