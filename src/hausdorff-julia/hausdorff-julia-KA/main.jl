using CUDA,KernelAbstractions
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(a,b,output,na,nb,slot)
 i=@index(Global,Linear)
 if i<=na
  @inbounds point=a[i]
  closest=floatmax(Float32)
  for j in 1:nb
   @inbounds other=b[j]
   dx=point[1]-other[1];dy=point[2]-other[2]
   closest=min(closest,dx*dx+dy*dy)
  end
  @inbounds bits=output[slot]
  while closest>reinterpret(Float32,bits)
   next=reinterpret(UInt32,closest)
   previous=(Atomix.@atomicreplace output[slot] bits=>next).old
   previous==bits&&break
   bits=previous
  end
 end
end
function run!(a,b,out,na,nb)
 kernel!(get_backend(out),256)(a,b,out,na,nb,Int32(1);ndrange=na)
 kernel!(get_backend(out),256)(b,a,out,nb,na,Int32(2);ndrange=nb)
 return
end
study(run!)
