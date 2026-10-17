using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,j,indices,counts,scores,template,threshold,priors,topk)
 ii=Int32(i)-Int32(1);tid=mod(ii,Int32(256));group=(Int32(j)-Int32(1))*Int32(CLASSES)+ii÷Int32(256);bins=JACC.shared(template)
 for part in Int32(0):Int32(7);@inbounds bins[part*Int32(256)+tid+1]=Int32(0);end
 JACC.sync_workgroup();histogram_body(tid,group,scores,bins,threshold,priors);JACC.sync_workgroup();suffix_body(tid,bins)
 if tid==0;@inbounds counts[group+1]=Int32(0);end
 JACC.sync_workgroup();select_body(tid,group,scores,indices,counts,bins,threshold,priors,topk)
 return
end
run!(indices,counts,scores,tmp,threshold,priors,topk)=(JACC.parallel_for(JACC.launch_spec(threads=(256,1),blocks=(CLASSES,BATCH),shmem_size=8192,sync=false),(256CLASSES,BATCH),kernel!,indices,counts,scores,tmp,threshold,priors,topk);nothing)
study(run!)
