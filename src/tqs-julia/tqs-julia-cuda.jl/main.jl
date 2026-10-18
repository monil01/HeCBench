using CUDA
include("../tqs_device.jl")
function taskqueue!(queue,data,consumed,it,offset,Q,T)
 tid=Int32(threadIdx().x);shared=@cuStaticSharedMem(Int32,3)
 if tid==1
  n=CUDA.atomic_add!(pointer(consumed,1),Int32(1));@inbounds shared[1]=n;shared[2]=queue[n+1].id;shared[3]=queue[n+1].op
 end
 sync_threads()
 @inbounds while shared[1]<Q
  id=shared[2];op=shared[3];idx=(id-offset)*T+tid
  if op==1
   for j in Int32(1):it;data[idx]+=T;end
   data[idx]+=id
  elseif op==2
   for j in Int32(1):Int32(1);data[idx]+=T;end
   data[idx]+=id
  end
  sync_threads()
  if tid==1
   n=CUDA.atomic_add!(pointer(consumed,1),Int32(1));shared[1]=n;shared[2]=queue[n+1].id;shared[3]=queue[n+1].op
  end
  sync_threads()
 end
 return
end
launch(c,offset)=(@cuda threads=c.T blocks=c.B taskqueue!(c.tasks.data,c.gpu.data,c.counter.data,Int32(c.it),offset,Int32(c.Q),Int32(c.T));nothing)
include("../study_common.jl")
tqs_study(launch)
