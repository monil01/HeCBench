using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function nhwckernel!(t,outer,x,y,template)
 shared=JACC.shared(template)
 if t<=32;@inbounds shared[t]=x[(outer-1)*32+t];end
 JACC.sync_workgroup()
 if t<=32
  c=t-1
  @inbounds y[(outer-1)*32+t]=shared[mod(c,4)*8+div(c,4)+1]
 end
 return
end
function nchwkernel!(t,batch,channel,x,y)
 n=batch-1;c=channel-1;offset=t-1
 if offset<64
  @inbounds y[(n*32+c)*64+offset+1]=x[(n*32+mod(c,4)*8+div(c,4))*64+offset+1]
 end
 return
end
function run!(phase,x,y,template)
 if phase==1
  JACC.parallel_for(JACC.launch_spec(threads=(256,1),blocks=(1,256),shmem_size=128,sync=false),(256,256),nhwckernel!,x,y,template)
 else
  JACC.parallel_for(JACC.launch_spec(threads=(256,1,1),blocks=(1,4,32),shmem_size=0,sync=false),(256,4,32),nchwkernel!,x,y)
 end
 return
end
study(run!)
