using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(ax,ay,az,gx,gy,gz,charge,sz,out,template,::Val{M}) where M
 t=@index(Local,Linear);group=@index(Group,Linear);shared=@localmem Float32 (320,)
 index=(group-1)*64+t
 @inbounds x=gx[index];y=gy[index];z=gz[index]
 acc=(0f0,0f0,0f0,0f0);offset=Int32(0)
 while offset<129
  len=min(Int32(64),Int32(129)-offset)
  if t<=len
   @inbounds begin
    if M==0
     k=(t-1)*5;shared[k+1]=ax[offset+t];shared[k+2]=ay[offset+t];shared[k+3]=az[offset+t];shared[k+4]=charge[offset+t];shared[k+5]=sz[offset+t]
    else
     shared[t]=ax[offset+t];shared[t+len]=ay[offset+t];shared[t+2len]=az[offset+t];shared[t+3len]=charge[offset+t];shared[t+4len]=sz[offset+t]
    end
   end
  end
  @synchronize
  for i in Int32(1):len
   @inbounds begin
    if M==0
     k=(i-1)*5;a=shared[k+1];b=shared[k+2];c=shared[k+3];q=shared[k+4];s=shared[k+5]
    else
     a=shared[i];b=shared[i+len];c=shared[i+2len];q=shared[i+3len];s=shared[i+4len]
    end
   end
   acc=(acc[1]+term(x[1],y[1],z[1],a,b,c,q,s,Val(M)),acc[2]+term(x[2],y[2],z[2],a,b,c,q,s,Val(M)),acc[3]+term(x[3],y[3],z[3],a,b,c,q,s,Val(M)),acc[4]+term(x[4],y[4],z[4],a,b,c,q,s,Val(M)))
  end
  @synchronize
  offset+=len
 end
 @inbounds out[index]=acc
end
run!(args...)=kernel!(get_backend(args[9]),64)(args...;ndrange=256)
study(run!)
