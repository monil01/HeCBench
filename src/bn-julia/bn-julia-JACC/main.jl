using CUDA,JACC,Printf,Libdl,SHA
include("../bayesian_device.jl")
include("immutable_parent.jl")
# Compiled immutable literal, not an extra device input allocation.
# JACC.shared performs512zero stores+oneinitialbarrier before our own overwrite.
const BN_SHARED_ZERO=SVector{512,Float32}(ntuple(_->0f0,Val(512)))
@inline function score_kernel!(i,sizepernode,localscore,data,lg)
 bn_gen_score!(Int32(i-1),sizepernode,localscore,data,lg)
 return nothing
end
@inline function compute_kernel!(i,taskperthr,sizepernode,localscore,mask,node,total,scores,parents)
 @inbounds begin
  id=Int32(i-1);tid=rem(id,Int32(256));bid=div(id,Int32(256))
  best,par=bn_jacc_thread_best(id,taskperthr,sizepernode,localscore,mask,node,total)
  shared=JACC.shared(BN_SHARED_ZERO);shared[tid+1]=best;shared[256+tid+1]=Float32(tid)
  JACC.sync_workgroup();stride=Int32(128)
  while stride>=1
   if tid<stride
    if shared[tid+stride+1]>shared[tid+1] || (shared[tid+stride+1]==shared[tid+1]&&shared[256+tid+stride+1]<shared[256+tid+1])
     shared[tid+1]=shared[tid+stride+1];shared[256+tid+1]=shared[256+tid+stride+1]
    end
   end
   JACC.sync_workgroup();stride=div(stride,Int32(2))
  end
  if tid==0;scores[bid+1]=shared[1];end
  if tid==Int32(shared[257]);for k in 1:4;parents[bid*4+k]=par[k];end;end
 end
 return nothing
end
function launch_score!(S,locals,data,lg)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 JACC.parallel_for(spec,(div(S,256)+1)*256,score_kernel!,S,locals,data,lg)
end
function launch_compute!(S,locals,mask,node,total,scores,parents)
 spec=JACC.launch_spec(;threads=256,shmem_size=2048,sync=false)
 JACC.parallel_for(spec,(div(total,256)+1)*256,compute_kernel!,Int32(1),S,locals,mask,node,total,scores,parents)
end
include("../study_common.jl")
bn_study(launch_score!,launch_compute!)
