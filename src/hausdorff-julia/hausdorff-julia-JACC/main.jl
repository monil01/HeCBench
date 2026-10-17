using CUDA
import JACC
JACC.@init_backend
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,a,b,output,na,nb,slot)
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
 return
end
function run!(a,b,out,na,nb)
 JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(na,256),shmem_size=0,sync=false),na,kernel!,a,b,out,na,nb,Int32(1))
 JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(nb,256),shmem_size=0,sync=false),nb,kernel!,b,a,out,nb,na,Int32(2))
 return
end
study(run!)
