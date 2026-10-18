using CUDA,KernelAbstractions
function callback end
include("../study_common.jl")
@kernel function compute_cost(P,C,W,T,S,n,d,x,K)
 i=Int32(@index(Global,Linear)-1);l=@index(Local,Linear)
 shared=@localmem Float32 (8,)
 if l==1
  for j=Int32(0):d-Int32(1);@inbounds shared[j+Int32(1)]=C[j*n+x+Int32(1)];end
 end
 @synchronize
 if i<n
  v=0f0
  for j=Int32(0):d-Int32(1)
   @inbounds a=C[j*n+i+Int32(1)]-shared[j+Int32(1)]
   v=muladd(a,a,v)
  end
  @inbounds p=P[i+Int32(1)];v*=p.weight;cur=p.cost;base=i*(K+Int32(1))
  if v<cur
   @inbounds S[i+Int32(1)]=UInt8(49);W[base+K+Int32(1)]=v-cur
  else
   @inbounds slot=T[Int32(p.assign)+Int32(1)];W[base+slot+Int32(1)]+=cur-v
  end
 end
end
const KERNEL=compute_cost(CUDABackend(),256)
function callback(pp,cp,wp,tp,sp,n,d,x,K)
 d==8||error("NamedfixtureDIM8")
 arrays=wrapped_inputs(pp,cp,wp,tp,sp,n,d,K)
 CUDA.stream!(CUDA.default_stream())do
  KERNEL(arrays...,Int32(n),Int32(d),Int32(x),Int32(K);ndrange=Int(n))
 end
 nothing
end
main()
