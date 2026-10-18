using CUDA,JACC,Atomix
include("../tqs_device.jl")
function taskqueue!(i,queue,data,consumed,it,offset,Q,T,template)
 tid=rem(Int32(i)-Int32(1),T)+Int32(1);shared=JACC.shared(template)
 if tid==1
  n=(Atomix.@atomic :monotonic consumed[1]+Int32(1))[1]
  @inbounds shared[1]=n;shared[2]=queue[n+1].id;shared[3]=queue[n+1].op
 end
 JACC.sync_workgroup()
 @inbounds while shared[1]<Q
  id=shared[2];op=shared[3];idx=(id-offset)*T+tid
  if op==1
   for j in Int32(1):it;data[idx]+=T;end
   data[idx]+=id
  elseif op==2
   for j in Int32(1):Int32(1);data[idx]+=T;end
   data[idx]+=id
  end
  JACC.sync_workgroup()
  if tid==1
   n=(Atomix.@atomic :monotonic consumed[1]+Int32(1))[1]
   shared[1]=n;shared[2]=queue[n+1].id;shared[3]=queue[n+1].op
  end
  JACC.sync_workgroup()
 end
 nothing
end
const template=CUDA.zeros(Int32,3)
function launch(c,offset)
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=c.T,blocks=c.B,shmem_size=12,sync=false)
 JACC.parallel_for(taskqueue!,spec,c.T*c.B,c.tasks.data,c.gpu.data,c.counter.data,Int32(c.it),offset,Int32(c.Q),Int32(c.T),template)
 nothing
end
include("../study_common.jl")
tqs_study(launch)
