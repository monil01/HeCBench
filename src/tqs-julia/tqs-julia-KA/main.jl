using CUDA,KernelAbstractions,Atomix
include("../tqs_device.jl")
@kernel unsafe_indices=true function taskqueue!(queue,data,consumed,it,offset,Q,T)
 tid=@index(Local,Linear);shared=@localmem Int32 (3,)
 if tid==1
  n=(Atomix.@atomic :monotonic consumed[1]+Int32(1))[1]
  @inbounds shared[1]=n;shared[2]=queue[n+1].id;shared[3]=queue[n+1].op
 end
 @synchronize
 @inbounds while shared[1]<Q
  id=shared[2];op=shared[3];idx=(id-offset)*T+tid
  if op==1
   for j in Int32(1):it;data[idx]+=T;end
   data[idx]+=id
  elseif op==2
   for j in Int32(1):Int32(1);data[idx]+=T;end
   data[idx]+=id
  end
  @synchronize
  if tid==1
   n=(Atomix.@atomic :monotonic consumed[1]+Int32(1))[1]
   shared[1]=n;shared[2]=queue[n+1].id;shared[3]=queue[n+1].op
  end
  @synchronize
 end
end
function launch(c,offset)
 taskqueue!(CUDABackend(),c.T)(c.tasks.data,c.gpu.data,c.counter.data,Int32(c.it),offset,Int32(c.Q),Int32(c.T);ndrange=c.T*c.B)
 nothing
end
include("../study_common.jl")
tqs_study(launch)
