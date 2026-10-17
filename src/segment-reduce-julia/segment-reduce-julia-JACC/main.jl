using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(index,input,sums,keys,s,template)
 t=mod(index-1,256)+1;g=div(index-1,256)+1;shared=JACC.shared(template)
 sum=Int32(0);j=t
 while j<=s
  @inbounds sum+=input[(g-1)*s+j]
  j+=256
 end
 @inbounds shared[t]=sum
 JACC.sync_workgroup()
 offset=128
 while offset>0
  if t<=offset;@inbounds shared[t]+=shared[t+offset];end
  JACC.sync_workgroup()
  offset÷=2
 end
 if t==1
  @inbounds sums[g]=shared[1];keys[g]=Int32(g-1)
 end
 return
end
run!(x,y,k,s,groups,template)=(JACC.parallel_for(JACC.launch_spec(threads=256,blocks=groups,shmem_size=1024,sync=false),groups*256,kernel!,x,y,k,s,template);nothing)
study(run!)
