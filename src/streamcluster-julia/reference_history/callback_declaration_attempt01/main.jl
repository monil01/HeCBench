using CUDA
include("../study_common.jl")
function compute_cost(P,C,W,T,S,n,d,x,K)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);l=threadIdx().x
 shared=CuDynamicSharedArray(Float32,Int(d))
 if l==1
  for j=Int32(0):d-Int32(1);@inbounds shared[j+Int32(1)]=C[j*n+x+Int32(1)];end
 end
 sync_threads()
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
 nothing
end
function callback(pp,cp,wp,tp,sp,n,d,x,K)
 arrays=wrapped_inputs(pp,cp,wp,tp,sp,n,d,K)
 @cuda threads=256 blocks=cld(n,256) shmem=4*d stream=CUDA.default_stream() compute_cost(arrays...,Int32(n),Int32(d),Int32(x),Int32(K))
 nothing
end
main()
