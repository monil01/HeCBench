using CUDA,KernelAbstractions
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(indices,src,data,n,::Val{T},::Val{OP}) where{T,OP}
 i=@index(Global,Linear)
 if i<=n
  @inbounds j=indices[i]+1
  @inbounds v=src[i]
  @inbounds bits=data[j]
  while true
   old=reinterpret(T,bits);want=applyop(old,v,Val(OP));next=reinterpret(typeof(bits),want)
   previous=(Atomix.@atomicreplace data[j] bits=>next).old
   previous==bits&&break
   bits=previous
  end
 end
end
run!(idx,src,out,n,t,op)=(kernel!(get_backend(out),256)(idx,src,out,n,t,op;ndrange=n);nothing)
study(run!)
