using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(input,sums,keys,s)
 t=Int(threadIdx().x);g=Int(blockIdx().x);shared=CuStaticSharedArray(Int32,256)
 sum=Int32(0);j=t
 while j<=s
  @inbounds sum+=input[(g-1)*s+j]
  j+=256
 end
 @inbounds shared[t]=sum
 CUDA.sync_threads()
 offset=128
 while offset>0
  if t<=offset;@inbounds shared[t]+=shared[t+offset];end
  CUDA.sync_threads()
  offset÷=2
 end
 if t==1
  @inbounds sums[g]=shared[1];keys[g]=Int32(g-1)
 end
 return
end
run!(x,y,k,s,groups,template)=(@cuda threads=256 blocks=groups kernel!(x,y,k,s);nothing)
study(run!)
