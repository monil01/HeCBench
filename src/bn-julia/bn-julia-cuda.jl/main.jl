using CUDA,Printf,Libdl,SHA
include("../bayesian_device.jl")
function score_kernel!(sizepernode,localscore,data,lg)
 id=Int32((blockIdx().x-1)*256+threadIdx().x-1)
 bn_gen_score!(id,sizepernode,localscore,data,lg)
 return nothing
end
function compute_kernel!(taskperthr,sizepernode,localscore,mask,node,total,scores,parents)
 @inbounds begin
  tid=Int32(threadIdx().x-1);bid=Int32(blockIdx().x-1);id=bid*Int32(256)+tid
  best,par=bn_thread_best(id,taskperthr,sizepernode,localscore,mask,node,total)
  shared=CuDynamicSharedArray(Float32,512);shared[tid+1]=best;shared[256+tid+1]=Float32(tid)
  sync_threads();stride=Int32(128)
  while stride>=1
   if tid<stride
    if shared[tid+stride+1]>shared[tid+1] || (shared[tid+stride+1]==shared[tid+1]&&shared[256+tid+stride+1]<shared[256+tid+1])
     shared[tid+1]=shared[tid+stride+1];shared[256+tid+1]=shared[256+tid+stride+1]
    end
   end
   sync_threads();stride=div(stride,Int32(2))
  end
  if tid==0;scores[bid+1]=shared[1];end
  if tid==Int32(shared[257]);for i in 1:4;parents[bid*4+i]=par[i];end;end
 end
 return nothing
end
function launch_score!(S,locals,data,lg)
 @cuda threads=256 blocks=div(S,256)+1 score_kernel!(S,locals,data,lg)
end
function launch_compute!(S,locals,mask,node,total,scores,parents)
 @cuda threads=256 blocks=div(total,256)+1 shmem=2048 compute_kernel!(Int32(1),S,locals,mask,node,total,scores,parents)
end
include("../study_common.jl")
bn_study(launch_score!,launch_compute!)
