using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function nhwckernel!(x,y)
 t=@index(Local,Linear);outer=@index(Group,Linear)
 shared=@localmem Float32 (32,)
 if t<=32;@inbounds shared[t]=x[(outer-1)*32+t];end
 @synchronize
 if t<=32
  c=t-1
  @inbounds y[(outer-1)*32+t]=shared[mod(c,4)*8+div(c,4)+1]
 end
end
@kernel function nchwkernel!(x,y)
 t=@index(Local,Linear);sb,batch,channel=@index(Group,NTuple)
 n=batch-1;c=channel-1;offset=(sb-1)*256+t-1
 if offset<64
  @inbounds y[(n*32+c)*64+offset+1]=x[(n*32+mod(c,4)*8+div(c,4))*64+offset+1]
 end
end
function run!(phase,x,y,template)
 if phase==1
  nhwckernel!(get_backend(y),256)(x,y;ndrange=256*256)
 else
  nchwkernel!(get_backend(y),(256,1,1))(x,y;ndrange=(256,4,32))
 end
 return
end
study(run!)
