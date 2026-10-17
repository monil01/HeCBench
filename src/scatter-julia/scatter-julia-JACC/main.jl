using CUDA
import JACC
JACC.@init_backend
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,indices,src,data,n,::Val{T},::Val{OP}) where{T,OP}
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
 return
end
run!(idx,src,out,n,t,op)=(JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false),n,kernel!,idx,src,out,n,t,op);nothing)
study(run!)
