using CUDA
function softmax_cuda!(input,output)
 tid=Int32(threadIdx().x)-Int32(1);row=Int32(blockIdx().x)-Int32(1)
 scratch=CuStaticSharedArray(Float32,256)

 base=row*Int32(384);v=-floatmax(Float32)
 for e=tid:Int32(256):Int32(383)
  @inbounds v=max(v,input[base+e+1])
 end
 @inbounds scratch[tid+1]=v
 CUDA.sync_threads()
 for offset in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<offset
   @inbounds scratch[tid+1]=max(scratch[tid+1],scratch[tid+offset+1])
  end
  CUDA.sync_threads()
 end
 @inbounds m=scratch[1]
 CUDA.sync_threads()
 v=0f0
 for e=tid:Int32(256):Int32(383)
  @inbounds v+=exp(input[base+e+1]-m)
 end
 @inbounds scratch[tid+1]=v
 CUDA.sync_threads()
 for offset in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tid<offset
   @inbounds scratch[tid+1]+=scratch[tid+offset+1]
  end
  CUDA.sync_threads()
 end
 @inbounds norm=1f0/scratch[1]
 for e=tid:Int32(256):Int32(383)
  @inbounds output[base+e+1]=exp(input[base+e+1]-m)*norm
 end
 return
end
function topk_cuda!(probs,weights,indices,source_rows,k::Int32)
 tid=Int32(threadIdx().x)-Int32(1);row=Int32(blockIdx().x)-Int32(1)
 values=CuStaticSharedArray(Float32,256);keys=CuStaticSharedArray(Int32,256)

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
  CUDA.sync_threads()
  for offset in (Int32(128),Int32(64),Int32(32),Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
   if tid<offset
    @inbounds av=values[tid+1];@inbounds bv=values[tid+offset+1]
    @inbounds ak=Int32(keys[tid+1]);@inbounds bk=Int32(keys[tid+offset+1])
    if bv>av || (bv==av && bk<ak)
     @inbounds values[tid+1]=bv
     @inbounds keys[tid+1]=bk
    end
   end
   CUDA.sync_threads()
  end
  if tid==Int32(0)
   @inbounds weights[row*k+rank+1]=values[1]
   @inbounds indices[row*k+rank+1]=Int32(keys[1])
   @inbounds source_rows[row*k+rank+1]=rank*Int32(257)+row
  end
  CUDA.sync_threads()
 end
 return
end
