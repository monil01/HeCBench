using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(input,sums,keys,s)
 t=@index(Local,Linear);g=@index(Group,Linear);shared=@localmem Int32 (256,)
 sum=Int32(0);j=t
 while j<=s
  @inbounds sum+=input[(g-1)*s+j]
  j+=256
 end
 @inbounds shared[t]=sum
 @synchronize
 offset=128
 while offset>0
  if t<=offset;@inbounds shared[t]+=shared[t+offset];end
  @synchronize
  offset÷=2
 end
 if t==1
  @inbounds sums[g]=shared[1];keys[g]=Int32(g-1)
 end
end
run!(x,y,k,s,groups,template)=(kernel!(get_backend(y),256)(x,y,k,s;ndrange=groups*256);nothing)
study(run!)
