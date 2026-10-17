using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel unsafe_indices=true function softmax_ka!(input,output)
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1)
 scratch=@localmem Float32 (256,)

 base=row*Int32(384);v=-floatmax(Float32)
 for e=tid:Int32(256):Int32(383)
  @inbounds v=max(v,input[base+e+1])
 end
 @inbounds scratch[tid+1]=v
 @synchronize
 for offset in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<offset
   @inbounds scratch[tid+1]=max(scratch[tid+1],scratch[tid+offset+1])
  end
  @synchronize
 end
 @inbounds m=scratch[1]
 @synchronize
 v=0f0
 for e=tid:Int32(256):Int32(383)
  @inbounds v+=exp(input[base+e+1]-m)
 end
 @inbounds scratch[tid+1]=v
 @synchronize
 for offset in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<offset
   @inbounds scratch[tid+1]+=scratch[tid+offset+1]
  end
  @synchronize
 end
 @inbounds norm=1f0/scratch[1]
 for e=tid:Int32(256):Int32(383)
  @inbounds output[base+e+1]=exp(input[base+e+1]-m)*norm
 end
 nothing
end
@kernel unsafe_indices=true function topk_ka!(probs,weights,indices,source_rows,k::Int32)
 tid=Int32(@index(Local,Linear))-Int32(1);row=Int32(@index(Group,Linear))-Int32(1)
 values=@localmem Float32 (256,);keys=@localmem Int32 (256,)

 base=row*Int32(384)
 for rank=Int32(0):k-Int32(1)
  best=-1f0;key=Int32(0)
  for expert=tid:Int32(256):Int32(383)
   prior=false
   for j=Int32(0):rank-Int32(1)
    @inbounds prior|=indices[row*k+j+1]==expert
   end
   if !prior
    @inbounds value=probs[base+expert+1]
    if value>best || (value==best && expert<key);best=value;key=expert;end
   end
  end
  @inbounds values[tid+1]=best
  @inbounds keys[tid+1]=key
  @synchronize
  for offset in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
   if tid<offset
    @inbounds av=values[tid+1];@inbounds bv=values[tid+offset+1]
    @inbounds ak=Int32(keys[tid+1]);@inbounds bk=Int32(keys[tid+offset+1])
    if bv>av || (bv==av && bk<ak)
     @inbounds values[tid+1]=bv
     @inbounds keys[tid+1]=bk
    end
   end
   @synchronize
  end
  if tid==Int32(0)
   @inbounds weights[row*k+rank+1]=values[1]
   @inbounds indices[row*k+rank+1]=Int32(keys[1])
   @inbounds source_rows[row*k+rank+1]=rank*Int32(257)+row
  end
  @synchronize
 end
 nothing
end
const BACKEND=CUDA.CUDABackend()
function launch!(a,p,w,i,r,k)
 softmax_ka!(BACKEND,256)(a,p;ndrange=65792)
 topk_ka!(BACKEND,256)(p,w,i,r,Int32(k);ndrange=65792)
 nothing
end
moe_study(launch!)
