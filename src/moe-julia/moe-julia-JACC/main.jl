using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function softmax_jacc!(index,input,output,template)
 linear=Int32(index)-Int32(1);tid=linear%Int32(256);row=linear÷Int32(256)
 scratch=JACC.shared(template)

 base=row*Int32(384);v=-floatmax(Float32)
 for e=tid:Int32(256):Int32(383)
  @inbounds v=max(v,input[base+e+1])
 end
 @inbounds scratch[tid+1]=v
 JACC.sync_workgroup()
 for offset in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<offset
   @inbounds scratch[tid+1]=max(scratch[tid+1],scratch[tid+offset+1])
  end
  JACC.sync_workgroup()
 end
 @inbounds m=scratch[1]
 JACC.sync_workgroup()
 v=0f0
 for e=tid:Int32(256):Int32(383)
  @inbounds v+=exp(input[base+e+1]-m)
 end
 @inbounds scratch[tid+1]=v
 JACC.sync_workgroup()
 for offset in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<offset
   @inbounds scratch[tid+1]+=scratch[tid+offset+1]
  end
  JACC.sync_workgroup()
 end
 @inbounds norm=1f0/scratch[1]
 for e=tid:Int32(256):Int32(383)
  @inbounds output[base+e+1]=exp(input[base+e+1]-m)*norm
 end
 nothing
end
@inline function topk_jacc!(index,probs,weights,indices,source_rows,k::Int32,template)
 linear=Int32(index)-Int32(1);tid=linear%Int32(256);row=linear÷Int32(256)
 arena=JACC.shared(template);values=arena;keys=@view arena[257:512]
 # Shared Float32 keys are exact for named384experts; globalindices remainInt32.

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
  JACC.sync_workgroup()
  for offset in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
   if tid<offset
    @inbounds av=values[tid+1];@inbounds bv=values[tid+offset+1]
    @inbounds ak=Int32(keys[tid+1]);@inbounds bk=Int32(keys[tid+offset+1])
    if bv>av || (bv==av && bk<ak)
     @inbounds values[tid+1]=bv
     @inbounds keys[tid+1]=bk
    end
   end
   JACC.sync_workgroup()
  end
  if tid==Int32(0)
   @inbounds weights[row*k+rank+1]=values[1]
   @inbounds indices[row*k+rank+1]=Int32(keys[1])
   @inbounds source_rows[row*k+rank+1]=rank*Int32(257)+row
  end
  JACC.sync_workgroup()
 end
 nothing
end
const TEMPLATE_SOFT=CUDA.zeros(Float32,256)
const TEMPLATE_TOP=CUDA.zeros(Float32,512)
const SPEC_SOFT=JACC.launch_spec(threads=256,blocks=257,shmem_size=1024,sync=false)
const SPEC_TOP=JACC.launch_spec(threads=256,blocks=257,shmem_size=2048,sync=false)
function launch!(a,p,w,i,r,k)
 JACC.parallel_for(SPEC_SOFT,65792,softmax_jacc!,a,p,TEMPLATE_SOFT)
 JACC.parallel_for(SPEC_TOP,65792,topk_jacc!,p,w,i,r,Int32(k),TEMPLATE_TOP)
 nothing
end
moe_study(launch!)
