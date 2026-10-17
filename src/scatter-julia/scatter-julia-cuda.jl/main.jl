using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(indices,src,data,n,::Val{T},::Val{OP}) where{T,OP}
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if i<=n
  @inbounds j=indices[i]+1
  @inbounds v=src[i]
  @inbounds bits=data[j]
  while true
   old=reinterpret(T,bits);want=applyop(old,v,Val(OP));next=reinterpret(typeof(bits),want)
   previous=CUDA.atomic_cas!(pointer(data,j),bits,next)
   previous==bits&&break
   bits=previous
  end
 end
 return
end
run!(idx,src,out,n,t,op)=(@cuda threads=256 blocks=cld(n,256) kernel!(idx,src,out,n,t,op);nothing)
study(run!)
