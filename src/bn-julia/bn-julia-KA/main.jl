using CUDA,KernelAbstractions,Printf,Libdl,SHA
include("../bayesian_device.jl")
include("immutable_parent.jl")
@kernel function score_kernel!(sizepernode,localscore,data,lg)
 id=Int32(@index(Global,Linear))-Int32(1)
 bn_gen_score!(id,sizepernode,localscore,data,lg)
end
@kernel function compute_kernel!(taskperthr,sizepernode,localscore,mask,node,total,scores,parents)
 @inbounds begin
  tid=Int32(@index(Local,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);id=bid*Int32(256)+tid
  best,par=bn_ka_thread_best(id,taskperthr,sizepernode,localscore,mask,node,total)
  shared=@localmem Float32 (512,);shared[tid+1]=best;shared[256+tid+1]=Float32(tid)
  @synchronize;stride=Int32(128)
  while stride>=1
   if tid<stride
    if shared[tid+stride+1]>shared[tid+1] || (shared[tid+stride+1]==shared[tid+1]&&shared[256+tid+stride+1]<shared[256+tid+1])
     shared[tid+1]=shared[tid+stride+1];shared[256+tid+1]=shared[256+tid+stride+1]
    end
   end
   @synchronize;stride=div(stride,Int32(2))
  end
  if tid==0;scores[bid+1]=shared[1];end
  if tid==Int32(shared[257]);for i in 1:4;parents[bid*4+i]=par[i];end;end
 end
end
function launch_score!(S,locals,data,lg)
 score_kernel!(get_backend(locals),256)(S,locals,data,lg;ndrange=(div(S,256)+1)*256)
end
function launch_compute!(S,locals,mask,node,total,scores,parents)
 compute_kernel!(get_backend(locals),256)(Int32(1),S,locals,mask,node,total,scores,parents;ndrange=(div(total,256)+1)*256)
end
include("../study_common.jl")
bn_study(launch_score!,launch_compute!)
